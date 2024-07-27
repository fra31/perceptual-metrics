import torch
from peft import LoraConfig, get_peft_model, PeftModel
from vision_transformer import vit_base
from vit_wrapper import ViTConfig, ViTModel


key_mapping = {
    'class_embedding': 'cls_token',
    'positional_embedding': 'pos_embed',
    'conv1.weight': 'patch_embed.proj.weight',
    'conv1.bias': 'patch_embed.proj.bias',
    'ln_pre.weight': 'pos_drop.weight',
    'ln_pre.bias': 'pos_drop.bias',
    'ln_post.weight': 'norm.weight',
    'ln_post.bias': 'norm.bias',
    'proj': 'proj'
}


class QuickGELU(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def openclip_to_dino(openclip_state_dict):
    # Initialize a new state dict for the dino-vit model
    new_state_dict = {}
    already_matched = set()
    # Iterate through each block to map the keys
    num_blocks = 12  # Assuming there are 12 blocks
    for i in range(num_blocks):
        new_state_dict = {}
        already_matched = set()
        num_blocks = 12  # Assuming there are 12 blocks
        for i in range(num_blocks):
            block_keys = [
                (f'transformer.resblocks.{i}.ln_1.weight', f'blocks.{i}.norm1.weight'),
                (f'transformer.resblocks.{i}.ln_1.bias', f'blocks.{i}.norm1.bias'),
                (f'transformer.resblocks.{i}.attn.in_proj_weight', f'blocks.{i}.attn.qkv.weight'),
                (f'transformer.resblocks.{i}.attn.in_proj_bias', f'blocks.{i}.attn.qkv.bias'),
                (f'transformer.resblocks.{i}.attn.out_proj.weight', f'blocks.{i}.attn.proj.weight'),
                (f'transformer.resblocks.{i}.attn.out_proj.bias', f'blocks.{i}.attn.proj.bias'),
                (f'transformer.resblocks.{i}.ln_2.weight', f'blocks.{i}.norm2.weight'),
                (f'transformer.resblocks.{i}.ln_2.bias', f'blocks.{i}.norm2.bias'),
                (f'transformer.resblocks.{i}.mlp.c_fc.weight', f'blocks.{i}.mlp.fc1.weight'),
                (f'transformer.resblocks.{i}.mlp.c_fc.bias', f'blocks.{i}.mlp.fc1.bias'),
                (f'transformer.resblocks.{i}.mlp.c_proj.weight', f'blocks.{i}.mlp.fc2.weight'),
                (f'transformer.resblocks.{i}.mlp.c_proj.bias', f'blocks.{i}.mlp.fc2.bias'),
            ]
            for src_key, dst_key in block_keys:
                if src_key in openclip_state_dict:
                    tensor = openclip_state_dict[src_key]
                    new_state_dict[dst_key] = tensor
                    already_matched.add(src_key)
    # Map the remaining keys using the predefined key_mapping
    for k, v in openclip_state_dict.items():
        if k not in already_matched and k in key_mapping:
            new_key = key_mapping[k]
            new_state_dict[new_key] = v
        elif k not in already_matched:
            print(f"Key {k} not found in the mapping")
    new_state_dict['cls_token'] = new_state_dict['cls_token'].unsqueeze(0).unsqueeze(0)
    new_state_dict['pos_embed'] = new_state_dict['pos_embed'].unsqueeze(0)
    new_state_dict['patch_embed.proj.bias'] = torch.zeros(new_state_dict['patch_embed.proj.weight'].shape[0])
    return new_state_dict


def load_lora_models(enc, arch, ckpt_path, device='cpu'):
    
    if arch == 'convnext-base-w':
        target_modules = []
        for n, p in enc.named_modules():
            if 'fc1' in n or 'fc2' in n:
                target_modules.append(n)
        config = LoraConfig(
            r=16,
            lora_alpha=0.5,
            lora_dropout=0.3,
            bias='none',
            target_modules=target_modules,
            )
        extractor_model = get_peft_model(
            ViTModel(enc, ViTConfig()),  # Needs adapting dreamsim.
            config).to(device)

        model = PeftModel.from_pretrained(
            extractor_model, ckpt_path).to(device)
        proj = None
        
    elif arch in ['ViT-B-16', 'ViT-B-32']:
        state_dict = enc.state_dict()
        dino_sd = openclip_to_dino(state_dict)
        # try loading it into dino
        patch_size = 16 if '16' in arch else 32
        dino_vit = vit_base(patch_size=patch_size).to(device)
        dino_vit.pos_drop = torch.nn.LayerNorm(dino_vit.embed_dim)
        proj = dino_sd.pop('proj')
        dino_vit.load_state_dict(dino_sd, strict=True)

        # # GeLU -> QuickGeLU
        # for blk in dino_vit .blocks:
        #     blk.mlp.act = QuickGELU()

        # # LN eps 1e-6 -> 1e-5
        # for m in dino_vit.modules():
        #     if isinstance(m, torch.nn.LayerNorm):
        #         m.eps = 1e-5

        target_modules = ['qkv']
        config = LoraConfig(
            r=16,
            lora_alpha=0.5,
            lora_dropout=0.3,
            bias='none',
            target_modules=target_modules)
   
        extractor_model = get_peft_model(
            ViTModel(dino_vit, ViTConfig()), config).to(device)

        model = PeftModel.from_pretrained(
            extractor_model, ckpt_path).to(device)
        
        
    return model, proj