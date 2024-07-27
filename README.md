<div align="center">

<h3>Adversarially Robust CLIP Models Induce Better (Robust) Perceptual Metrics</h3>
</div>

---------------------------------
### Robust and NIGHTS fine-tuned perceptual models

#### Example usage

**2AFC tasks:** to test a model on the NIGHTS dataset (either split), one can use
```python
python3 eval.py \
	--shortname <model_id> \
	--split [test_no_imagenet | test_imagenet] --dataset nights \
	--n_ex -1 --batch_size 100 --device 'cuda:0' \
	--model_dir <model_dir> --data_dir <data_dir> \
	#
	# Attacks flags (skip for clean accuracy only).
	--norm Linf --eps 4 \
	--n_iter 100 --attack_name apgd \
	--n_restarts 1 --use_rs
```
with a `model_id` from the pre-trained models after downloading the relative checkpoint.

#### List of pre-trained models
	
| Model ID | Backbone      | Robust FT Method    | NIGHTS FT |    Checkpoint                             |
|----------|---------------|------------|-------------|-------------|
| `convnext_base_w` | ConvNeXt-B | -- | -- | HF |
| `convnext_base_w-fare` | ConvNeXt-B | FARE  | -- |[Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/GCPXBDEE5PoCngy)     |
| `convnext_base_w-tecoa` | ConvNeXt-B | TeCoA  | -- | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/zHKCC9aS7rf4qCt)     |
| `mlp-convnext_base_w-fare` | ConvNeXt-B | FARE | MLP  | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/Fb73e3i2PmWfwpN) |
| `mlp-convnext_base_w-tecoa` | ConvNeXt-B | TeCoA | MLP | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/2beeHj3DZNDbswZ) |
| `lora-convnext_base_w-fare` | ConvNeXt-B | FARE | LoRA | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/LxWHf7x9r3rXHPA) |
| `lora-convnext_base_w-tecoa` | ConvNeXt-B | TeCoA | LoRA | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/f3dJ44YJkeY47Bj)|
| `vit-b-16` | ViT-B/16 | -- | -- | HF |
| `vit-b-16-fare` | ViT-B/16 | FARE  | -- | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/fg7JHQzASiNnxCg)     |
| `vit-b-16-tecoa` | ViT-B/16 | TeCoA  | -- | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/ZBkmbMrAwgfeeSa)     |
| `mlp-vit-b-16-fare` | ViT-B/16 | FARE | MLP  | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/yYSM3pd7acJGZRq)     |
| `mlp-vit-b-16-tecoa` | ViT-B/16 | TeCoA | MLP  | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/BLL8c8DbBxX8RsB)     |
| `lora-vit-b-16-fare` | ViT-B/16 | FARE | LoRA  | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/jNwtQKK3oareL83)     |
| `lora-vit-b-16-tecoa` | ViT-B/16 | TeCoA | LoRA  | [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/6aG2kPbpqCjodic)     |


