import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from autoattack.other_utils import Logger, makedir
import time
import json
import sys

import utils_perceptual_models
import utils_perceptual_data
import utils_perceptual_eval
import attacks_clip



def get_model_and_preprocess(args, **kwargs):
    """Load model and pre-processing to use."""

    model, preprocess = utils_perceptual_models.get_model_and_transforms(
        modelname=args.modelname,
        ckptpath=args.ckptpath,
        pretrained=args.pretrained,
        source=args.source,
        device=args.device,
        mlp_head=args.mlp_head,
        lora_weights=args.lora_weights,
        logger=kwargs['logger'],
        model_dir=args.model_dir,
        )
    model.eval()
    
    logger = kwargs.get('logger')

    if args.mlp_head is not None:
        # TODO: make this more general.
        #from dreamsim.model import MLP
        #mlp_path, fts = utils_perceptual_models.MLP_HEADS_DICT[args.mlp_head]
        mlp_path, fts = utils_perceptual_models.PRETRAINED_MODELS[args.shortname]['mlp_info']
        mlp_path = os.path.join(args.model_dir, mlp_path)
        logger.log(f'Loading MLP head from {mlp_path}.')
        #mlp = MLP(*fts)
        mlp = utils_perceptual_models.MLP(*fts)
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
            vis_enc = utils_perceptual_models.ClipVisionModel(model.visual, None, normalize_fn)
        else:
            import utils_lora
            #lora_path = utils_perceptual_models.LORA_WEIGHTS_DICT[args.lora_weights]
            lora_path = utils_perceptual_models.PRETRAINED_MODELS[args.shortname]['lora_path']
            lora_path = os.path.join(args.model_dir, lora_path)
            logger.log(f'Loading LoRA weights from {lora_path}.')
            # Following needed for using models fine-tuned as in DreamSim.
            lora_model, lora_proj = utils_lora.load_lora_models(
                model.visual,
                args.arch.replace('+lora', '').replace('+head', ''),
                lora_path)
            if lora_proj is not None:
                lora_model = utils_perceptual_models.ProjModel(lora_model, lora_proj)
            vis_enc = utils_perceptual_models.ClipVisionModel(lora_model, None, normalize_fn)
        vis_enc.eval()
        vis_enc.to(args.device)

        # if args.int_fts_subset is not None:
        #     # TODO: combine internal features with mlp head.
        #     logger.log((
        #         f'Using internal features: mode={args.int_fts_mode},'
        #         f' subset={args.int_fts_subset} pool_fn={args.int_fts_pool}'))
        #     int_fts_subset = {
        #         'out': 'out',
        #         'm1+out': [-1, 'out'],
        #         'm1': [-1,],
        #         'm7+out': [-7, 'out']
        #         }[args.int_fts_subset]
        #     fts_model = utils_perceptual_models.IntFeatWrapper(vis_enc, args.int_fts_mode)
        #     fp = lambda x: fts_model(x, fts=int_fts_subset, pool_fn=args.int_fts_pool,
        #                              output_normalize=False)
        # else:
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
    
    utils_perceptual_eval.resolve_args(args)
    logger = Logger(args.log_path)
    logger.log(args.log_path)
    makedir(args.savedir)

    # Get model.
    fp, preprocess = get_model_and_preprocess(args, logger=logger)
    print('Model loaded.')

    #sys.exit()

    # Load dataset.
    ds, loader = utils_perceptual_data.load_dataset(
        args, preprocess=preprocess, logger=logger)
    print('Data loaded.')

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

    # Collect results.
    x_ref = None
    if args.attack_name is not None:
        x_ref = [batch[0] for batch in loader]
        str_dets = utils_perceptual_eval.check_imgs_loader(x_adv, x_ref, args.norm)
        logger.log(str_dets)
    logger.log('clean')
    clean_acc = utils_perceptual_eval.get_acc(
        fp, loader, device=args.device, cust_im_ref=x_ref, n_ex=args.n_ex,
        mode=args.metric_type, logger=logger)
    if args.attack_name is not None:
        logger.log('adv')
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
            info[args.runinfo] = acc
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
    args = utils_perceptual_eval.get_args()
    main(args)
