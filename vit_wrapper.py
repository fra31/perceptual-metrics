"""Adapted from https://github.com/ssundaram21/dreamsim."""

from transformers import PretrainedConfig
from transformers import PreTrainedModel


class ViTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ViTModel(PreTrainedModel):
    config_class = ViTConfig

    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        if hasattr(model, 'blocks'):
            self.blocks = model.blocks
        elif hasattr(model, 'trunk'):
            self.blocks = model.trunk.stages  # For convnext.
        else:
            for n, _ in model.named_modules():
                print(n)
            raise ValueError('Unknown architecture.')

    def forward(self, x):
        return self.model(x)
