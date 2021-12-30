# mostly from https://github.com/facebookresearch/deit/blob/main/cait_models.py
# with some inspiration from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
# and https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cait.py
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.vision_transformer import Mlp


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, class_attention=False, talking_heads=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.class_attention = class_attention
        self.talking_heads = talking_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        if talking_heads:
            self.proj_l = nn.Linear(num_heads, num_heads)
            self.proj_w = nn.Linear(num_heads, num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape

        if self.class_attention:
            q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        if self.talking_heads:
            attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        if self.talking_heads:
            attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1 if self.class_attention else N, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls

class LayerScale(nn.Module):
    def __init__(self, dim, attentionBlock, mlp_ratio=4., drop=0., drop_path=0., init_values=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attentionBlock
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls: Optional[torch.Tensor] = None):
        if x_cls is not None:
            u = torch.cat((x_cls, x), dim=1)
            x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
            x = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    
class CaiT(nn.Module):
    def __init__(self,
                 num_classes=4,
                 num_heads=8,
                 embed_dim=1024,
                 depth=24,
                 depth_token_only=2,
                 mlp_ratio=4.,
                 mlp_ratio_token_only=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 init_scale=1e-5):
        super().__init__()
            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, max(1, num_patches), embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)] 
        self.blocks = nn.ModuleList([
            LayerScale(
                dim=embed_dim,
                attentionBlock=Attention(
                    embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    class_attention=False,
                    talking_heads=True,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate),
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                init_values=init_scale)
            for i in range(depth)])

        self.blocks_token_only = nn.ModuleList([
            LayerScale(
                dim=embed_dim,
                attentionBlock=Attention(
                    embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    class_attention=True,
                    talking_heads=False,
                    attn_drop=0.0,
                    proj_drop=0.0),
                mlp_ratio=mlp_ratio_token_only,
                drop=0.0,
                drop_path=0.0,
                init_values=init_scale)
            for i in range(depth_token_only)])
            
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # TODO: load pretrained weights partially
    #     if pretrained:
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             url="https://dl.fbaipublicfiles.com/deit/XXS24_224.pth",
    #             map_location="cpu", check_hash=True
    #         )
    #         checkpoint_no_module = {}
    #         for k in model.state_dict().keys():
    #             checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
    #
    #         model.load_state_dict(checkpoint_no_module)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)

        # TODO: should we have pos embedding?
        # x = x + self.pos_embed
        # x = self.pos_drop(x)
        x = x.mean(dim=1, keepdim=True)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x