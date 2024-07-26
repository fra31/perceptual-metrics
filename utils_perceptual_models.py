import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import open_clip
import clip
from PIL import Image
import os
from types import SimpleNamespace

import utils_perceptual_eval


SHORTNAMES = {
    None: 'openai',
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_True_imagenet_vit-l-sup-10k-3adv-lr1e-5_wd_1e-4_fFTvv_final.pt': 'sup-fFTvv',
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_False_vit-l-unsup-clean-0p1-lr1e-5_ZUSEW_final.pt': 'unsup-ZUSEW',
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_False_imagenet_vit-l-unsup-clean-0p0-lr1e-5-wd-1e-4_mCGle_final.pt': 'unsup-mCGle',
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_True_imagenet_vit-l-tecoa-eps4-2epoch_BaLvU_final.pt': 'tecoa-4-BaLvU',
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_True_imagenet_vit-l-tecoa-eps2-2epoch_OM5H5_final.pt': 'tecoa-2-OM5H5',
    '../robust-clip/ViT-L-14_openai_imagenet_txtSup_False_imagenet_vit-l-fareP-2epoch_qZvwS_final.pt': 'fare+4-qZvwS',
    '../robust-clip/ViT-L-14__mnt_cschlarmann37_project_multimodal_clip-finetune_ViT-L-14_openai_imagenet_txtSup_False_imagenet_vit-l-fareP-eps2-2epoch_17yHI_temp_checkpoints_fallback_16000.pt_imagenet_txtSup_False_imagenet_vit-l-fareP-eps2-2epoch_9JNNU_final.pt': 'fare+2-9JNNU',
    '../robust-clip/CLIP-ViT-B-32-DataComp.XL-s13B-b90K_none_imagenet_ce_imagenet_TECOA4_oGFNb.pt': 'tecoa-4-oGFNb',
    '../robust-clip/CLIP-ViT-B-32-DataComp.XL-s13B-b90K_none_imagenet_l2_imagenet_FARE4_t1gfj.pt': 'fare-4-t1gfj',
    'hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K': 'openclip',
    '../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K-original.pt': 'openclip-laion',
    '../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K_none_imagenet_ce_imagenet_TECOA4_9j55I.pt': 'tecoa-4-9j55I',
    '../robust-clip/CLIP-ViT-B-32-laion2B-s34B-b79K_none_imagenet_l2_imagenet_FARE4_dCUnU.pt': 'fare-4-dCUnU',
    '../lipsim/model.ckpt-1.pth': 'lipsim-pretr',
    '../lipsim/model.ckpt-435_margin=0.5.pth': 'lipsim-margin=.5',
    '../lipsim/model.ckpt-435_margin=0.2.pth': 'lipsim-margin=.2',
    '../robust-clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg_none_imagenet_ce_imagenet_TECOA4_cREDo.pt': 'tecoa-4-cREDo',
    '../robust-clip/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg_none_imagenet_l2_imagenet_FARE4_VFul3.pt': 'fare-4-VFul3',
    'hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg': 'openclip-convnext-base-w',
    '../robust-clip/CLIP-ViT-B-16-DataComp.XL-s13B-b90K.pt': 'openclip',
    '../robust-clip/CLIP-ViT-B-16-DataComp.XL-s13B-b90K_none_imagenet_ce_imagenet_TECOA4_LFNiy.pt': 'tecoa-4-LFNiy',
    '../robust-clip/CLIP-ViT-B-16-DataComp.XL-s13B-b90K_none_imagenet_l2_imagenet_FARE4_W1Hzc.pt': 'fare-4-W1Hzc',
    '../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K.pt': 'openclip-laion',
    '../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K_none_imagenet_ce_imagenet_TECOA4_mMRaV.pt': 'tecoa-4-mMRaV',
    '../robust-clip/CLIP-ViT-B-16-laion2B-s34B-b88K_none_imagenet_l2_imagenet_FARE4_SKirS.pt': 'fare-4-SKirS',
    '../R-LPIPS/checkpoints/latest_net_linf_ref.pth': 'r-lpips-linf-ref',
    'dreamsim:open_clip_vitb32': 'dreamsim-openclip',
    'dreamsim:clip_vitb32': 'dreamsim-clip',
    'dreamsim:dino_vitb16': 'dreamsim-dino',
    'dreamsim:ensemble': 'dreamsim',
    }

MLP_HEADS_DICT = {
    # convnext-b
    'tecoa-4-cREDo': (
        '../robust-clip/mlp_ours_cvnxtb_laion2B_tecoa4_embedding_lr0.0003_bs512_wd0.0_hidsize512_marg0.05_2024-03-01_15-33-06/lightning_logs/version_0/checkpoints/epoch=08.ckpt',
        (640, 512)
    ),
    'tecoa-4-mMRaV': (
        '../robust-clip/mlp_ours_vitb16_laion2B_tecoa4_embedding_lr0.0003_bs512_wd0.0_hidsize512_marg0.05_2024-03-01_09-43-52/lightning_logs/version_0/checkpoints/epoch=04.ckpt',
        (512, 512),
    ),
    'fare-4-VFul3': (
        '../robust-clip/mlp_ours_cvnxtb_laion2B_fare4_embedding_lr0.0003_bs512_wd0.0_hidsize512_marg0.05_2024-03-01_15-34-01/lightning_logs/version_0/checkpoints/epoch=04.ckpt',
        (640, 512),
    ),
    'fare-4-SKirS': (
        '../robust-clip/mlp_ours_vitb16_laion2B_fare4_embedding_lr0.0003_bs512_wd0.0_hidsize512_marg0.05_2024-03-01_11-17-05/lightning_logs/version_0/checkpoints/epoch=06.ckpt',
        (512, 512),
    ),
    'tecoa-4-9j55I': (
        '../robust-clip/mlp_ours_vitb32_laion2B_tecoa4_embedding_lr0.0003_bs512_wd0.0_hidsize512_marg0.05_2024-03-01_07-52-33/lightning_logs/version_0/checkpoints/epoch=02.ckpt',
        (512, 512),
    ),
    'fare-4-dCUnU': (
        '../robust-clip/mlp_ours_vitb32_laion2B_fare4_embedding_lr0.0003_bs512_wd0.0_hidsize512_marg0.05_2024-03-01_09-22-06/lightning_logs/version_0/checkpoints/epoch=06.ckpt',
        (512, 512),
    ),
    'openclip-convnext-base-w': (
        '../robust-clip/mlp_ours_cvnxtb_laion2B_original_embedding_lr0.0003_bs512_wd0.0_hidsize512_marg0.05_augmFalse_2024-03-04_14-39-08/lightning_logs/version_0/checkpoints/epoch=07.ckpt',
        (640, 512),
    ),
    'openclip-laion-vitb16': (
        '../robust-clip/mlp_ours_vitb16_laion2B_original_embedding_lr0.0003_bs512_wd0.0_hidsize512_marg0.05_augmFalse_2024-03-04_15-00-27/lightning_logs/version_0/checkpoints/epoch=09.ckpt',
        (512, 512),
    ),
    'openclip-laion-vitb32': (
        '../robust-clip/mlp_ours_vitb32_laion2B_original_embedding_lr0.0003_bs512_wd0.0_hidsize512_marg0.05_augmFalse_2024-03-04_14-56-42/lightning_logs/version_0/checkpoints/epoch=10.ckpt',
        (512, 512),
    ),
    # new naming
    'mlp-convnext_base_w-fare-eps4': ('mlp-convnext_base_w-fare-eps4.pth', (640, 512)),
}

LORA_WEIGHTS_DICT = {
    'tecoa-4-cREDo': '../robust-clip/lora_ours_cvnxtb_laion2B_tecoa4_embedding_lr0.0003_bs32_wd0.0_hidsize1_marg0.05_lorar16_loraalph0.5_loradrop0.3_2024-03-01_17-53-54/lightning_logs/version_0/checkpoints/epoch_6_convnext_base',
    'tecoa-4-mMRaV': '../robust-clip/lora_ours_vitb16_laion2B_tecoa4_embedding_lr0.0003_bs32_wd0.0_hidsize1_marg0.05_lorar16_loraalph0.5_loradrop0.3_2024-03-01_11-12-53/lightning_logs/version_0/checkpoints/epoch_5_open_clip_vitb16',
    'fare-4-VFul3': '../robust-clip/lora_ours_cvnxtb_laion2B_fare4_embedding_lr0.0003_bs32_wd0.0_hidsize1_marg0.05_lorar16_loraalph0.5_loradrop0.3_2024-03-01_17-54-20/lightning_logs/version_0/checkpoints/epoch_0_convnext_base/',
    'fare-4-SKirS': '../robust-clip/lora_ours_vitb16_laion2B_fare4_embedding_lr0.0003_bs32_wd0.0_hidsize1_marg0.05_lorar16_loraalph0.5_loradrop0.3_2024-03-01_11-17-15/lightning_logs/version_0/checkpoints/epoch_4_open_clip_vitb16',
    'tecoa-4-9j55I': '../robust-clip/lora_ours_vitb32_laion2B_tecoa4_embedding_lr0.0003_bs32_wd0.0_hidsize1_marg0.05_2024-03-01_07-32-14_lorar16_loraalph0.5_loradrop0.3/lightning_logs/version_0/checkpoints/epoch_5_open_clip_vitb32',
    'fare-4-dCUnU': '../robust-clip/lora_ours_vitb32_laion2B_fare4_embedding_lr0.0003_bs32_wd0.0_hidsize1_marg0.05_lorar16_loraalph0.5_loradrop0.3_2024-03-01_09-22-11/lightning_logs/version_0/checkpoints/epoch_3_open_clip_vitb32',
    'openclip-convnext-base-w': '../robust-clip/lora_ours_cvnxtb_laion2B_original_embedding_lr0.0003_bs32_wd0.0_hidsize1_marg0.05_augmFalse_lorar16_loraalph0.5_loradrop0.3_2024-03-05_10-19-57/lightning_logs/version_0/checkpoints/epoch_0_convnext_base',
    'openclip-laion-vitb16': '../robust-clip/lora_ours_vitb16_laion2B_original_embedding_lr0.0003_bs32_wd0.0_hidsize1_marg0.05_augmFalse_lorar16_loraalph0.5_loradrop0.3_2024-03-04_15-51-50/lightning_logs/version_0/checkpoints/epoch_1_open_clip_vitb16',
    'openclip-laion-vitb32': '../robust-clip/lora_ours_vitb32_laion2B_original_embedding_lr0.0003_bs32_wd0.0_hidsize1_marg0.05_augmFalse_lorar16_loraalph0.5_loradrop0.3_2024-03-04_16-03-59/lightning_logs/version_0/checkpoints/epoch_2_open_clip_vitb32'
}

PRETRAINED_MODELS = {
    'convnext_base-fare': {
        'ckptpath': 'convnext_base_w-fare-eps4.pt',
        },
    'mlp-convnext_base_w-fare': {
        'ckptpath': 'convnext_base_w-fare-eps4.pt',  # Backbone checkpoint.
        'mlp_head': 'mlp-convnext_base_w-fare-eps4',
        'mlp_info': ('mlp-convnext_base_w-fare-eps4.pt', (640, 512)),
    },
    'lora-convnext_base_w-fare': {
        'ckptpath': 'convnext_base_w-fare-eps4.pt',  # Backbone checkpoint.
        'lora_weights': 'lora-convnext_base_w-fare',
        'lora_path': 'lora-convnext_base_w-fare',
    },
}


class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, args, normalize, all_tokens=False, proj=True):
        super().__init__()
        self.model = model
        self.args = args
        self.normalize = normalize
        try:
            self.proj = model.proj
        except:
            print('Warning: no proj module.')
        if all_tokens:
            self.model.output_tokens = True
        if not proj:
            self.model.proj = None
            # todo also compare with proj (otherwise the proj layer is not trained)
            # note proj is only applied to class token
        # self.output_normalize = args.output_normalize

    def forward(self, vision_, output_normalize, **kwargs):
        embedding = self.model(self.normalize(vision_), **kwargs)
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)

        if hasattr(self.model, 'output_tokens') and self.model.output_tokens:
            # flatten and concatenate all tokens
            return torch.hstack([embedding[0].flatten(1), embedding[1].flatten(1)])
        else:
            return embedding


class ProjModel(nn.Module):
    def __init__(self, fts, proj):
        super().__init__()
        self.fts = fts
        self.proj = proj

    def forward(self, x, **kwargs):
        out = self.fts(x, **kwargs)
        return out @ self.proj.to(x.device)


# class IntFeatWrapper(torch.nn.Module):
    
#     def __init__(self, model, mode) -> None:
#         super().__init__()
#         self.model = model
#         self.mode = mode
#         self.vis = {}
#         utils_vis.track_layers(self.model.model, self.vis, self.mode)

#     def forward(self, x, fts='out', pool_fn=None, **kwargs):
#         y = self.model(x, **kwargs)
#         if fts == 'out': 
#             return y
#         else:
#             if pool_fn == 'avg':
#                 _pool_fn = lambda z: z.reshape([z.shape[0], z.shape[1], -1]).mean(dim=-1)
#             elif pool_fn == 'vec':
#                 _pool_fn = lambda z: z.reshape([z.shape[0], -1])
#             elif pool_fn == 'avg+norm':
#                 def _pool_fn(z):
#                     z = z.reshape([z.shape[0], z.shape[1], -1]).mean(dim=-1)
#                     return F.normalize(z, p=2, dim=1)
#             elif pool_fn == 'cls_tkn':
#                 def _pool_fn(z):
#                     if len(z.shape) == 2:
#                         return z
#                     return z[0]  # Use cls token embedding.
#             elif pool_fn == 'cls_tkn+norm':
#                 def _pool_fn(z):
#                     if len(z.shape) == 2:
#                         return F.normalize(z, p=2, dim=-1)
#                     return F.normalize(z[0], p=2, dim=-1)  # Use cls token embedding.
#             else:
#                 _pool_fn = lambda z: z

#             if isinstance(fts, int):
#                 z = list(self.vis.values())[fts]
#                 return _pool_fn(z)
#             elif isinstance(fts, (list, tuple)):
#                 l_fts = list(self.vis.values())
#                 #z = [_pool_fn(l_fts[i]) for i in fts]
#                 z = []
#                 for i in fts:
#                     if isinstance(i, int):
#                         z_curr = l_fts[i]
#                     elif i == 'out':
#                         z_curr = y
#                     z.append(_pool_fn(z_curr))
#                 return torch.cat(z, dim=-1)
#             else:
#                 raise ValueError(f'Unknown features name: {fts}.')


# Copied from https://github.com/ssundaram21/dreamsim/blob/main/dreamsim/model.py.
class MLP(torch.nn.Module):
    """
    MLP head with a single hidden layer and residual connection.
    """
    def __init__(self, in_features: int, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(in_features, self.hidden_size, bias=True)
        self.fc2 = torch.nn.Linear(self.hidden_size, in_features, bias=True)

    def forward(self, img):
        x = self.fc1(img)
        x = F.relu(x)
        return self.fc2(x) + img

    
def get_model_and_transforms(
        modelname='ViT-L-14', ckptpath=None, pretrained='openai', source='openclip',
        **kwargs):

    logger = kwargs['logger']
    cache_dir = kwargs.get('model_dir', './')

    if source == 'openclip':
        model, _, preprocess = open_clip.create_model_and_transforms(
            modelname, pretrained=pretrained, cache_dir=cache_dir,
            )
        if 'convnext_base_w' in modelname:  # The original model was trained at 256px.
            preprocess.transforms[1] = transforms.CenterCrop(224)
        mlp_head = kwargs.get('mlp_head', None)
        lora_weights = kwargs.get('lora_weights', None)
        if mlp_head is not None or lora_weights is not None:  # DreamSim doesn't use OpenCLIP preprocessing.
            logger.log('Using DreamSim preprocessing instead of OpenCLIP one.')
            preprocess = transforms.Compose([
                transforms.Resize((224, 224), 
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.ToTensor(),  # Will be removed.
            ])
        
    elif source == 'clip':
        model, preprocess = clip.load(modelname)

    elif source == 'lipsim':
        # Adapted from https://github.com/SaraGhazanfari/lipsim/tree/main.
        from lipsim.core.models.l2_lip.model import L2LipschitzNetwork

        config = SimpleNamespace()
        if modelname == 'convnet-small':
            config.depth = 20
            config.num_channels = 45
            config.depth_linear = 7 
            config.n_features = 1024
            config.conv_size = 5
        model = L2LipschitzNetwork(config, n_classes=1792).eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    elif source == 'r-lpips':
        # From https://github.com/SaraGhazanfari/R-LPIPS/tree/main, plus removing
        # global device and setting strict loading.
        import lpips

        model = lpips.LPIPSSimple(
            pretrained=True,
            net='alex',
            version='0.1',
            lpips=True,
            spatial=False,
            pnet_rand=False,
            pnet_tune=False,
            use_dropout=True,
            model_path=ckptpath,  # Already loads the checkpoint.
            eval_mode=True,
        )
        preprocess = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])  # Unclear what to use for datasets other than BAPPS.

    elif source == 'dreamsim':
        from dreamsim import dreamsim

        model, preprocess = dreamsim(
            pretrained=True,
            cache_dir=cache_dir,
            dreamsim_type=modelname,
            device=kwargs.get('device'),  # Changing device after initialization has to
                                          # be done manually for each model part.
            )
        preprocess = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])  # The preprocess loaded is a function.

    if ckptpath is not None:
        ckpt = torch.load(ckptpath, map_location='cpu')
        if source in ['openclip']:
            model.visual.load_state_dict(ckpt, strict=True)
        elif source == 'lipsim':
            model.load_state_dict(
                {k.replace('module.model.', ''): v for k, v in ckpt['model_state_dict'].items()},
                strict=True)

    return model, preprocess


# def adapt_pos_enc(model, key='positional_embedding', new_res=None, old_res=None,
#     patch_size=None):
#     """Interpolate the positional embedding of ViTs."""

#     if old_res is None:
#         old_res = model.image_size[0]
#     if patch_size is None:
#         patch_size = model.patch_size[0]

#     ckpt = model.state_dict()
    
#     print(ckpt[key].shape)
#     new_pos_enc = utils.interpolate_pos_encoding(
#         pos_embed=ckpt[key].unsqueeze(0),
#         new_img_size=new_res,
#         old_img_size=old_res,
#         patch_size=patch_size).squeeze(0)
#     print(new_pos_enc.shape)
#     ckpt[key] = new_pos_enc

#     model.positional_embedding = torch.nn.Parameter(
#         torch.randn(new_pos_enc.shape))
#     model.load_state_dict(ckpt)

#     model.image_size = (new_res, new_res)


# def get_data(dataset='imagenet', data_dir='/scratch/datasets/imagenet/',
#     prepr=None, n_ex=10, seed=0, image_paths=None):

#     if dataset == 'imagenet':
#         print(f'Fixing seed={seed}.')
#         torch.manual_seed(1)

#         test_dataset = datasets.ImageFolder(root=data_dir + 'val', transform=prepr)
#         test_loader = torch.utils.data.DataLoader(test_dataset,
#                                           batch_size=n_ex,
#                                           shuffle=True,
#                                           num_workers=16,
#                                           pin_memory=True,
#                                           )

#         x_test, y_test = next(iter(test_loader))

#     elif dataset == 'mmhal':
#         if image_paths is None:
#             image_paths = os.listdir(data_dir)
#             image_paths = [item for item in image_paths if item.endswith('.jpg')]
#             image_paths = image_paths[:n_ex]

#         x_test = []
#         for image_path in image_paths:
#             print(image_path)
#             with open(os.path.join(data_dir, image_path), 'rb') as f:
#                 img = Image.open(f)
#                 img = img.convert('RGB')
            
#             img = prepr(img)
#             x_test.append(img.unsqueeze(0))
#         x_test = torch.cat(x_test, 0)

#     else:
#         raise ValueError(f'Unknown dataset: {dataset}.')
    
#     print(x_test.shape, x_test.max(), x_test.min())
#     return x_test


# def no_axes(ax):
#     for x in ax.reshape([-1]):
#         x.axis('off')



# def main(args):

    # Get model.
    # model, preprocess = get_model_and_transforms(
    #     modelname=args.modelname, ckptpath=args.ckptpath)
    # model.eval()

    # # Adapt to different image resolution.
    # if args.img_res != 224:
    #     adapt_pos_enc(model.vision, key='positional_embedding', new_res=args.img_res,
    #         old_res=224)
    #     preprocess.transforms[:2] = [
    #         transforms.Resize(
    #             args.img_res,
    #             interpolation=transforms.InterpolationMode("bicubic")),
    #         transforms.CenterCrop(args.img_res),
    #         ]

    # normalize_fn = lambda x: x  # Just identity.
    # vis_enc = ClipVisionModel(model.visual, None, normalize_fn)
    # vis_enc.to(args.device)

    # # Get data.
    # x_test = get_data(dataset=args.dataset, data_dir=args.data_dir,
    #     prepr=preprocess, n_ex=args.n_ex, seed=args.seed, image_paths=None)

    # # Get maps.
    # all_maps = []
    # mode = utils_vis.INTERM_REPRS['CLIP']
    # print(f'Using {mode} as inner representation, layer {args.map_layer}.')
    
    # n_batches = math.ceil(args.n_ex / args.batch_size)
    # bs = args.batch_size

    # for i in range(n_batches):
    #     vis = {}
    #     utils_vis.track_layers(vis_enc.model.transformer, vis, mode)

    #     with torch.no_grad():
    #         x = x_test[bs * i:bs * (i + 1)]
    #         _ = vis_enc(x.to(device), output_normalize=False,)

    #     ks = list(vis.keys())
    #     all_maps.append(vis[ks[args.map_layer]][1].clone().cpu())

    # all_maps = torch.cat(all_maps, dim=0)

    # # TODO: complete this.
    # fname = None
    # torch.save(all_maps, fname)

    #print(args)


if __name__ == '__main__':
    
    pass
