import torch
import torch.nn as nn
from collections import OrderedDict

from clip.model import Transformer, LayerNorm

class FusionModel(nn.Module):
    def __init__(self, state_dict: OrderedDict, config):
        super().__init__()

        width = state_dict["visual.proj"].shape[1] # output_dim of ViT, which is 512 for ViT-B/16
        heads = width // 64
        layers = config.network.fusion_model_layers
        seq_length = config.data.seg_num * config.data.seg_length # sequence length (T)

        self.frame_positional_embedding = nn.Parameter(torch.randn(seq_length, width)) # (T, 512) # type: ignore 
        self.ln_final = LayerNorm(width)

        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads
        )

        print("=" * 64)
        print(f"Fusion Model: transformer_width: {width}, heads: {heads}, frame_num: {seq_length}")
        print("=" * 64)

        self.initialize_parameters()

    def initialize_parameters(self):
        # modified from CLIP initialize_parameters()
        nn.init.normal_(self.frame_positional_embedding, std=0.02)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)    # type: ignore
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)   # type: ignore
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)          # type: ignore
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)      # type: ignore
    
    def forward(self, x: torch.Tensor):
        x = x.contiguous() # (bs, T, 512)
        orig_dtype = x.dtype # orig_dtype: torch.float16

        x = x + self.frame_positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) # (bs, T, 512)

        return x.mean(dim=1, keepdim=False).to(dtype=orig_dtype) # (bs, 512)