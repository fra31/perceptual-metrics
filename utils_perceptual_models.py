import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import open_clip
import clip
import os
from types import SimpleNamespace


PRETRAINED_MODELS = {
    'convnext_base_w': {
        'ckptpath': 'hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg',
    },
    'convnext_base_w-fare': {
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
    'convnext_base_w-tecoa': {
        'ckptpath': 'convnext_base_w-tecoa-eps4.pt',
    },
    'mlp-convnext_base_w-tecoa': {
        'ckptpath': 'convnext_base_w-tecoa-eps4.pt',  # Backbone checkpoint.
        'mlp_head': 'mlp-convnext_base_w-tecoa-eps4',
        'mlp_info': ('mlp-convnext_base_w-tecoa-eps4.pt', (640, 512)),
    },
    'lora-convnext_base_w-tecoa': {
        'ckptpath': 'convnext_base_w-tecoa-eps4.pt',  # Backbone checkpoint.
        'lora_weights': 'lora-convnext_base_w-tecoa',
        'lora_path': 'lora-convnext_base_w-tecoa',
    },
    'vit-b-16': {
        'ckptpath': 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
    },
    'vit-b-16-fare': {
        'ckptpath': 'vit-b-16-fare-eps4.pt',
    },
    'mlp-vit-b-16-fare': {
        'ckptpath': 'vit-b-16-fare-eps4.pt',  # Backbone checkpoint.
        'mlp_head': 'mlp-vit-b-16-fare-eps4',
        'mlp_info': ('mlp-vit-b-16-fare-eps4.pt', (512, 512)),
    },
    'lora-vit-b-16-fare': {
        'ckptpath': 'vit-b-16-fare-eps4.pt',  # Backbone checkpoint.
        'lora_weights': 'lora-vit-b-16-fare',
        'lora_path': 'lora-vit-b-16-fare',
    },
    'vit-b-16-tecoa': {
        'ckptpath': 'vit-b-16-tecoa-eps4.pt',
    },
    'mlp-vit-b-16-tecoa': {
        'ckptpath': 'vit-b-16-tecoa-eps4.pt',  # Backbone checkpoint.
        'mlp_head': 'mlp-vit-b-16-tecoa-eps4',
        'mlp_info': ('mlp-vit-b-16-tecoa-eps4.pt', (512, 512)),
    },
    'lora-vit-b-16-tecoa': {
        'ckptpath': 'vit-b-16-tecoa-eps4.pt',  # Backbone checkpoint.
        'lora_weights': 'lora-vit-b-16-tecoa',
        'lora_path': 'lora-vit-b-16-tecoa',
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
    """Add projection layer for LoRA models (following DreamSim)."""
    def __init__(self, fts, proj):
        super().__init__()
        self.fts = fts
        self.proj = proj

    def forward(self, x, **kwargs):
        out = self.fts(x, **kwargs)
        return out @ self.proj.to(x.device)


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

    else:
        raise ValueError(f'Unknown source: {source}.')

    if ckptpath is not None:
        ckpt = torch.load(ckptpath, map_location='cpu')
        if source in ['openclip']:
            model.visual.load_state_dict(ckpt, strict=True)

    return model, preprocess


if __name__ == '__main__':
    print('Available models:')
    for k in PRETRAINED_MODELS.keys():
        print(k)
    pass
