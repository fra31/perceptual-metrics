import torch
import torch.nn.functional as F
import argparse
import os

import utils_perceptual_models


def get_args():

    parser = argparse.ArgumentParser(description='')
    # model
    parser.add_argument('--ckptpath', type=str)
    parser.add_argument('--shortname', type=str)
    parser.add_argument('--modelname', type=str)
    parser.add_argument('--mlp_head', type=str)
    parser.add_argument('--lora_weights', type=str)
    #parser.add_argument('--int_fts_pool', type=str)
    #parser.add_argument('--int_fts_subset', type=str)
    parser.add_argument('--model_dir', type=str, default='./')
    # data
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--n_ex', type=int, default=8)
    parser.add_argument('--split', type=str, default='test_imagenet')
    # attacks
    parser.add_argument('--attack_name', type=str)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--eps', type=float, default=8.)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--alpha_init', type=float)
    parser.add_argument('--use_rs', action='store_true')
    parser.add_argument('--n_restarts', type=int, default=1)
    # others
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--map_layer', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_to_dict', action='store_true')
    #parser.add_argument('--save_preds', action='store_true')
    #parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savedir', type=str, default='./results_clip')
    parser.add_argument('--logdir', type=str, default='./logs_clip')
    parser.add_argument('--dictname', type=str)

    args = parser.parse_args()
    
    return args


def cosine_sim(u, v):
    """Compute cosine similarity."""
    
    if len(u.shape) == 1:
        u = u.unsqueeze(0)
    if len(v.shape) == 1:
        v = v.unsqueeze(0)
    
    u = F.normalize(u, p=2, dim=1)
    v = F.normalize(v, p=2, dim=1)
    return (u * v).sum(dim=-1)


def get_emb(model, loader, device, cust_im_ref=None, n_ex=-1, bs=None,):
    """Get embedding for 2AFC data."""

    data = []
    #im_ref = []

    if cust_im_ref is not None:
        print(f'Using non-default reference images!')
    
    with torch.no_grad():
        
        if bs is None:
            seen_imgs = 0

            for i, (im_ref, im_left, im_right, lab, id) in enumerate(loader):
                if cust_im_ref is not None:
                    im_ref = cust_im_ref[i]
                if len(im_ref.shape) == 3:
                    im_ref = im_ref.unsqueeze(0)
                    im_left = im_left.unsqueeze(0)
                    im_right = im_right.unsqueeze(0)

                emb_ref = model(im_ref.to(device)).cpu()
                emb_left = model(im_left.to(device)).cpu()
                emb_right = model(im_right.to(device)).cpu()
                data.append((emb_ref, emb_left, emb_right, lab, id))

                seen_imgs += im_ref.shape[0]
                if n_ex != -1 and seen_imgs >= n_ex:
                    break

        else:
            print(f'Using batch size={bs}.')

            if n_ex == -1:
                n_ex = len(loader)
            x_test = []
            x_left = []
            x_right = []
            labs = []
            ids = []
            seen_imgs = 0

            for i, (im_ref, im_left, im_right, lab, id) in enumerate(loader):
                x_test.append(im_ref)
                x_left.append(im_left)
                x_right.append(im_right)
                labs.append(lab)
                ids.append(id)

                if (i + 1) % bs == 0 or i + 1 == n_ex:
                    idx = torch.arange(seen_imgs, i + 1).long()
                    #print(idx)
                    emb_ref = model(torch.stack(x_test, dim=0).to(device)).cpu()
                    emb_left = model(torch.stack(x_left, dim=0).to(device)).cpu()
                    emb_right = model(torch.stack(x_right, dim=0).to(device)).cpu()
                    labs = torch.tensor(labs)
                    ids = torch.tensor(ids)
                    data.append((emb_ref, emb_left, emb_right, labs, ids))

                    seen_imgs += len(x_test)
                    x_test = []
                    x_left = []
                    x_right = []
                    labs = []
                    ids = []

                if i + 1 == n_ex:
                    break

    return data


def get_sim(outputs, logger=None, verbose=True):
    """Compute accuracy on preferences."""

    acc = 0.
    stats = []
    l_acc = []
    n_ex = 0
    
    for emb_ref, emb_left, emb_right, lab, id in outputs:
        sim_left = cosine_sim(emb_ref, emb_left).squeeze(0)
        sim_right = cosine_sim(emb_ref, emb_right).squeeze(0)
        pred = (sim_right > sim_left).float()  # Labels represent whether the right image has more votes.
        stats.append((id, lab, pred))
        l_acc.append(pred == lab)
        acc += (pred == lab).float().sum()
        n_ex += emb_ref.shape[0]

    dets = f'acc={acc / n_ex:.2%} ({acc:.0f}/{n_ex})'
    if verbose:
        if logger is not None:
            logger.log(dets)
        else:
            print(dets)

    try:
        acc_dets = torch.stack(l_acc, dim=0)
    except:
        acc_dets = torch.cat(l_acc, dim=0)

    return stats, (acc.item() / n_ex, acc.item(), n_ex), acc_dets


@torch.no_grad()
def get_acc(model, loader, device, cust_im_ref=None, n_ex=-1, mode='lpips',
            logger=None):

    if mode == 'lpips':
        acc = 0
        seen_imgs = 0
        for i, (im_ref, im_left, im_right, lab, _) in enumerate(loader):
            if cust_im_ref is not None:
                im_ref = cust_im_ref[i]
            if len(im_ref.shape) == 3:
                im_ref = im_ref.unsqueeze(0)
                im_left = im_left.unsqueeze(0)
                im_right = im_right.unsqueeze(0)
            y = lpips_clf(model, im_ref.to(device), im_left.to(device),
                          im_right.to(device), normalize=True)
            pred = y.max(-1)[1].cpu()
            acc += (pred == lab).float().sum()
            seen_imgs += im_ref.shape[0]

            if n_ex != -1 and seen_imgs >= n_ex:
                break
        
        logger.log(f'acc={acc / seen_imgs:.2%} ({acc:.0f}/{seen_imgs})')
        return (acc.item() / seen_imgs, acc.item(), seen_imgs)

    elif mode == 'embedding':
        output_clean = get_emb(
            model, loader, device=device, cust_im_ref=cust_im_ref, n_ex=n_ex)
        _, clean_acc, _ = get_sim(output_clean, logger)
        return clean_acc
    
    elif mode in ['odd-one-out', 'odd-one-out-lpips']:
        acc = 0
        seen_imgs = 0
        for i, (im0, im1, im2, lab, _) in enumerate(loader):
            assert len(im0.shape) == 4
            pred = odd_one_out_clf(
                model, [im0.to(device), im1.to(device), im2.to(device)],
                mode='lpips' if 'lpips' in mode else 'embedding')
            acc += (pred.cpu() == lab).float().sum()
            seen_imgs += im0.shape[0]

            if n_ex != -1 and seen_imgs >= n_ex:
                break
        
        logger.log(f'acc={acc / seen_imgs:.2%} ({acc:.0f}/{seen_imgs})')
        return (acc.item() / seen_imgs, acc.item(), seen_imgs)

    else:
        raise ValueError(f'Unknown mode: {mode}.')


def clf(model, x, emb_left, emb_right):

    emb_ref = model(x)
    sim_left = cosine_sim(emb_ref, emb_left)
    sim_right = cosine_sim(emb_ref, emb_right)
    logits = torch.stack((sim_left, sim_right), dim=1)
    return logits


def lpips_clf(model, x_ref, x0, x1, normalize=True):
    """LPIPS-based classifier."""

    # TODO: features of non-reference images could be saved to
    # save 2 forward passes.
    y0 = model(x_ref, x0, normalize=normalize).squeeze()
    y1 = model(x_ref, x1, normalize=normalize).squeeze()
    logits = torch.stack((y0, y1), dim=1)
    return -logits  # Predict image with minimum distance.


def odd_one_out_clf(model, xs=[], mode='embedding'):
    """Classifier for odd-one-out tasks with 3 images."""

    # TODO: return logits instread of predictions.

    assert len(xs) == 3, 'Only 3 images supported.'

    if mode == 'embedding':
        embs = [model(x) for x in xs]

        # Compute similarity of pairs and predict the image which is not in the one
        # with highest similarity.
        sim01 = cosine_sim(embs[0], embs[1])
        sim02 = cosine_sim(embs[0], embs[2])
        sim12 = cosine_sim(embs[1], embs[2])

    elif mode == 'lpips':
        # Negative LPIPS distance as we need similarity.
        sim01 = -model(xs[0], xs[1], normalize=True).squeeze()
        sim02 = -model(xs[0], xs[2], normalize=True).squeeze()
        sim12 = -model(xs[1], xs[2], normalize=True).squeeze()

    preds = torch.zeros(xs[0].shape[0], device=xs[0].device).long()
    idx2 = (sim01 > sim02) * (sim01 > sim12)
    preds[idx2] = 2
    idx1 = (sim02 > sim01) * (sim02 > sim12)
    preds[idx1] = 1
    return preds


def check_imgs_loader(x_adv, x_clean, norm):

    nans = 0
    mins = 1e10
    maxs = -1e10
    max_diff = 0.

    for adv, x in zip(x_adv, x_clean):

        delta = (adv - x).view(adv.shape[0], -1)
        if norm == 'Linf':
            res = delta.abs().max(dim=1)[0]
        elif norm == 'L2':
            res = (delta ** 2).sum(dim=1).sqrt()
        elif norm == 'L1':
            res = delta.abs().sum(dim=1)

        max_diff = max(max_diff, res.max().item())
        nans += (adv != adv).sum()
        maxs = max(maxs, adv.max().item())
        mins = min(mins, adv.min().item())

    str_det = 'max {} pert: {:.5f}, nan in imgs: {}, max in imgs: {:.5f}, min in imgs: {:.5f}'.format(
        norm, max_diff, nans, maxs, mins)
    print(str_det)
    
    return str_det


def resolve_args(args):

    #args.shortname = utils_perceptual_models.SHORTNAMES.get(args.ckptpath, 'unknown')
    model_info = utils_perceptual_models.PRETRAINED_MODELS.get(args.shortname, None)
    if model_info is None:
        raise ValueError(f'Unknown model: {args.shortname}')
    args.ckptpath = model_info['ckptpath']
    args.mlp_head = model_info.get('mlp_head', None)
    args.lora_weights = model_info.get('lora_weights', None)

    if (args.ckptpath is not None and not args.ckptpath.startswith('hf-hub') \
        and not args.ckptpath.startswith('dreamsim')):  # The other models are automatically downloaded.
        args.ckptpath = os.path.join(args.model_dir, args.ckptpath)
    
    # Not very clear, but to use more HF checkpoints.
    # TODO: improve this.
    args.pretrained = 'openai' if args.ckptpath is None else None
    args.source = 'openclip'
    args.arch = None
    args.metric_type = 'embedding'
    if args.modelname is None:  # Detect backbone from checkpoint name.
        assert args.ckptpath is not None, 'Backbone type needed for OpenAI models.'
        if 'ViT-L-14' in args.ckptpath:
            args.modelname = 'ViT-L-14'
        elif 'ViT-B-32' in args.ckptpath or 'vitb32' in args.ckptpath:
            args.modelname = 'ViT-B-32'
        elif ('ViT-B-16' in args.ckptpath or 'vitb16' in args.ckptpath
              or 'vit-b-16' in args.ckptpath):
            args.modelname = 'ViT-B-16'
        elif 'convnext_base_w' in args.ckptpath:
            args.arch = 'convnext-base-w'
            args.modelname = 'hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg'
            args.pretrained = None
            if args.modelname == args.ckptpath:  # Use pre-trained model.
                args.ckptpath = None
        elif 'lipsim' in args.ckptpath:
            args.modelname = 'convnet-small'
        elif 'R-LPIPS' in args.ckptpath:
            args.modelname = 'alexnet'
            args.metric_type = 'lpips'
        elif args.ckptpath == 'dreamsim:ensemble':
            args.modelname = 'ensemble'
        else:
            raise ValueError('Unknown architecture.')
    if args.arch is None:
        args.arch = args.modelname  # For logging.
    if (args.ckptpath is not None and args.ckptpath.startswith('hf-hub') \
        and args.arch != 'convnext-base-w'):
        args.modelname = args.ckptpath
        args.pretrained = None
        args.ckptpath = None
    if args.ckptpath is not None and args.ckptpath.startswith('dreamsim:'):
        args.modelname = args.ckptpath.replace('dreamsim:', '')
        args.pretrained = None
        args.ckptpath = None
        args.source = 'dreamsim'
    if args.mlp_head is not None:
        args.arch += f'+head'
        #args.shortname += f'+{args.mlp_head}'  # This should be in the name already now.
    if args.lora_weights is not None:
        args.arch += f'+lora'
        #args.shortname += f'+{args.lora_weights}'  # This should be in the name already now.
    if args.ckptpath is not None:
        if 'lipsim' in args.ckptpath:
            args.source = 'lipsim'
        if 'R-LPIPS' in args.ckptpath:
            args.source = 'r-lpips'
    if args.dataset == 'things':
        args.metric_type = 'odd-one-out' if args.source != 'r-lpips' else 'odd-one-out-lpips'

    if args.norm == 'Linf':
        args.eps /= 255.
    args.logdir = f'{args.logdir}/{args.dataset}_{args.split}'
    os.makedirs(args.logdir, exist_ok=True)
    args.log_path = (
        #f'{args.logdir}/log_{args.arch}_{args.shortname}'
        f'{args.logdir}/log_{args.shortname}'
        f'_n_ex={args.n_ex}'
        )
    args.runinfo = None
    if args.attack_name is not None:
        runinfo = (
            f'{args.attack_name}-{args.loss}-{args.n_iter}x{args.n_restarts}'
            f'_{args.norm}_eps={args.eps:.5f}_alpha={args.alpha_init}'
            f'_rs={args.use_rs}')
        args.log_path += '_' + runinfo
        args.runinfo = runinfo
    args.log_path += '.txt'
