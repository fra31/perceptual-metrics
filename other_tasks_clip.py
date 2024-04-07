import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset
from torchvision import transforms
#import pandas as pd
#import numpy as np
#from PIL import Image
import os
#from autoattack import AutoAttack
from autoattack.other_utils import L1_norm, L2_norm, L0_norm, Logger, makedir
import time
import json

import vis_attn_clip
import utils_perceptual_data
import utils_perceptual_eval
import attacks_clip
import utils_vis


class ProbeModel(nn.Module):
    def __init__(self, fts, head):
        super().__init__()
        self.fts = fts
        self.head = head

    def forward(self, x, **kwargs):
        out = self.fts(x, **kwargs)
        return self.head(out)


class ProjModel(nn.Module):
    def __init__(self, fts, proj):
        super().__init__()
        self.fts = fts
        self.proj = proj

    def forward(self, x, **kwargs):
        out = self.fts(x, **kwargs)
        return out @ self.proj.to(x.device)


def get_model_and_preprocess(args, **kwargs):
    """Load model and pre-processing to use."""

    model, preprocess = vis_attn_clip.get_model_and_transforms(
        modelname=args.modelname,
        ckptpath=args.ckptpath,
        pretrained=args.pretrained,
        source=args.source,
        device=args.device,
        mlp_head=args.mlp_head,
        lora_weights=args.lora_weights,
        logger=kwargs['logger'],
        )
    model.eval()
    
    logger = kwargs.get('logger')

    if args.mlp_head is not None:
        # TODO: make this more general.
        from dreamsim.model import MLP
        mlp_path, fts = vis_attn_clip.MLP_HEADS_DICT[args.mlp_head]
        logger.log(f'Loading MLP head from {mlp_path}.')
        mlp = MLP(*fts)
        if mlp_path is not None:
            ckpt_mlp = torch.load(mlp_path, map_location='cpu')
            mlp.load_state_dict(
                {k.replace('perceptual_model.mlp.', ''): v for k, v in ckpt_mlp['state_dict'].items()})
        mlp.eval()
        mlp.to(args.device)

    if args.source in ['openclip', 'clip']:
        # Move normalization to FP.
        norm_layer = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711))
        preprocess.transforms = preprocess.transforms[:-1]
        normalize_fn = norm_layer

        if args.lora_weights is None:
            vis_enc = vis_attn_clip.ClipVisionModel(model.visual, None, normalize_fn)
        else:
            import utils_lora
            #enc = ProbeModel(model.visual, mlp)
            lora_path = vis_attn_clip.LORA_WEIGHTS_DICT[args.lora_weights]
            logger.log(f'Loading LoRA weights from {lora_path}.')
            # Following needed for using models fine-tuned as in DreamSim.
            lora_model, lora_proj = utils_lora.load_lora_models(
                model.visual,
                args.arch.replace('+lora', '').replace('+head', ''),
                lora_path)
            if lora_proj is not None:
                lora_model = ProjModel(lora_model, lora_proj)
            vis_enc = vis_attn_clip.ClipVisionModel(lora_model, None, normalize_fn)
        vis_enc.eval()
        vis_enc.to(args.device)

        if args.int_fts_subset is not None:
            # TODO: combine internal features with mlp head.
            logger.log((
                f'Using internal features: mode={args.int_fts_mode},'
                f' subset={args.int_fts_subset} pool_fn={args.int_fts_pool}'))
            int_fts_subset = {
                'out': 'out',
                'm1+out': [-1, 'out'],
                'm1': [-1,],
                'm7+out': [-7, 'out']
                }[args.int_fts_subset]
            fts_model = vis_attn_clip.IntFeatWrapper(vis_enc, args.int_fts_mode)
            fp = lambda x: fts_model(x, fts=int_fts_subset, pool_fn=args.int_fts_pool,
                                     output_normalize=False)
        else:
            if args.mlp_head is None:
                fp = lambda x: vis_enc(x, output_normalize=False)
            else:
                def fp(x):
                    out = vis_enc(x, output_normalize=False)
                    return mlp(out)

    elif args.source in ['lipsim']:
        model.eval()
        model.to(args.device)
        fp = lambda x: model(x)

    elif args.source in ['r-lpips']:
        model.eval()
        model.to(args.device)
        fp = lambda *x, **y: model(*x, **y)

    elif args.source == 'dreamsim':
        model.eval()
        fp = lambda x: model.embed(x)

    return fp, preprocess











def main(args):
    
    args.shortname = vis_attn_clip.SHORTNAMES.get(args.ckptpath, 'unknown')
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
        elif 'ViT-B-16' in args.ckptpath or 'vitb16' in args.ckptpath:
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
        args.shortname += f'+{args.mlp_head}'
    if args.lora_weights is not None:
        args.arch += f'+lora'
        args.shortname += f'+{args.lora_weights}'
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
    makedir(args.logdir)
    args.log_path = (
        f'{args.logdir}/log_{args.arch}_{args.shortname}'
        f'_n_ex={args.n_ex}'
        )
    if args.int_fts_subset is not None:
        args.int_fts_mode = utils_vis.INTERM_REPRS[args.arch]
        args.log_path += f'_fts={args.int_fts_subset}-{args.int_fts_pool}'
    if args.attack_name is not None:
        runinfo = (
            f'{args.attack_name}-{args.loss}-{args.n_iter}x{args.n_restarts}'
            f'_{args.norm}_eps={args.eps:.5f}_alpha={args.alpha_init}'
            f'_rs={args.use_rs}')
        args.log_path += '_' + runinfo
    args.log_path += '.txt'
    logger = Logger(args.log_path)
    logger.log(args.log_path)
    makedir(args.savedir)

    # Get model.
    fp, preprocess = get_model_and_preprocess(args, logger=logger)

    # Load dataset.
    ds, loader = utils_perceptual_data.load_dataset(
        args, preprocess=preprocess, logger=logger)

    # Run attacks.
    startt = time.time()

    if args.attack_name in ['apgd', 'apgd-largereps', 'square']:
        # TODO: merge in one attack.
        if args.n_restarts == 1:
            x_adv, acc = attacks_clip.eval_loader(
                fp,
                loader, #ds,
                n_ex=args.n_ex,
                bs=args.batch_size,
                device=args.device,
                norm=args.norm,
                eps=args.eps,
                loss=args.loss,
                n_iter=args.n_iter,
                use_rs=args.use_rs, 
                attack_name=args.attack_name,
                log_path=args.log_path,
                alpha_init=args.alpha_init,
                metric_type=args.metric_type,
            )
        else:
            x_adv, acc = attacks_clip.eval_restarts_loader(
                fp,
                loader, #ds,
                n_ex=args.n_ex,
                bs=args.batch_size,
                device=args.device,
                norm=args.norm,
                eps=args.eps,
                loss=args.loss,
                n_iter=args.n_iter,
                use_rs=True, 
                attack_name=args.attack_name,
                log_path=args.log_path,
                alpha_init=args.alpha_init,
                metric_type=args.metric_type,
                n_restarts=args.n_restarts,
                attacks=[f'apgd-{args.loss}']
            )
    
    elif args.attack_name is None:
        logger.log('Only clean accuracy.')

    else:
        raise ValueError(f'Unknown attack: {args.attack_name}.')

    totalt = time.time() - startt
    logger.log(f'Attack time: {totalt:.1f} s')

    # Put images in single batches.
    # x_ref = []
    # x_left = []
    # x_right = []
    # for i, (r, a, b, _, _) in enumerate(ds):
    #     x_ref.append(r)
    #     #x_left.append(a)
    #     #x_right.append(b)
    #     if i + 1 == args.n_ex:
    #         break
    # x_ref = torch.stack(x_ref, dim=0)
    #x_left = torch.stack(x_left, dim=0)
    #x_right = torch.stack(x_right, dim=0)

    # Collect results.
    x_ref = None
    if args.attack_name is not None:
        x_ref = [batch[0] for batch in loader]
        str_dets = utils_perceptual_eval.check_imgs_loader(x_adv, x_ref, args.norm)
        logger.log(str_dets)
    logger.log('clean')
    # output_clean = utils_perceptual_eval.get_emb(
    #     fp, loader, device=args.device, cust_im_ref=x_ref, n_ex=args.n_ex)
    # _, clean_acc, _ = utils_perceptual_eval.get_sim(output_clean, logger)
    clean_acc = utils_perceptual_eval.get_acc(
        fp, loader, device=args.device, cust_im_ref=x_ref, n_ex=args.n_ex,
        mode=args.metric_type, logger=logger)
    if args.attack_name is not None:
        logger.log('adv')
        # output_adv = utils_perceptual_eval.get_emb(
        #     fp, loader, device=args.device, cust_im_ref=x_adv, n_ex=args.n_ex)
        # _, acc, _ = utils_perceptual_eval.get_sim(output_adv, logger)
        acc = utils_perceptual_eval.get_acc(
            fp, loader, device=args.device, cust_im_ref=x_adv, n_ex=args.n_ex,
            mode=args.metric_type, logger=logger)

    # Save results.
    if args.save_to_dict:
        dictname = 'acc_dets.json' if args.dictname is None else args.dictname
        dictname = os.path.join(args.savedir, dictname)
        if not os.path.exists(dictname):  # Create empty dict if not existing.
            with open(dictname, 'w') as f:
                json.dump(dict(), f)
        # Results dict.
        modelid = f'{args.arch}_{args.shortname}'
        dsname = f'{args.dataset}_{args.split}_n_ex={args.n_ex}'
        info = {'clean': clean_acc}
        if args.attack_name is not None:
            info[runinfo] = acc
        # Add results.
        with open(dictname, 'r') as f:
            dets = json.load(f)
        if modelid not in dets.keys():
            dets[modelid] = {}
        if dsname not in dets[modelid].keys():
            dets[modelid][dsname] = {}
        for k, v in info.items():
            while k in dets[modelid][dsname].keys() and k != 'clean':
                k += '_new'
            dets[modelid][dsname][k] = v
        with open(dictname, 'w') as f:
            json.dump(dets, f, indent='\t')


if __name__ == '__main__':
    args = vis_attn_clip.get_args()
    main(args)
