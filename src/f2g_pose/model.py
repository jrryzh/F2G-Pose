# Derived from Uni6d and PoinTr/AdaPoinTr; see THIRD_PARTY.md.
from functools import partial
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .transformer import *
from . import geometry as misc
from .geometry import compute_rotation_matrix_from_ortho6d
from .chamfer import ChamferDistanceL1
from .radio import load_frozen_radio

class SelfAttnBlockApi(nn.Module):
    """
        1. Norm Encoder Block
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'
            combine_style = 'onebyone'
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat', k=10, n_group=2, num_experts=4, top_k=2, use_moe=False):
        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat', 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        if use_moe:
            self.mlp = MoeMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), num_experts=num_experts, top_k=top_k, act_layer=act_layer, drop=drop)
        else:
            self.mlp = WideMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), num_experts=num_experts, top_k=top_k, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))
        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0, qkv_bias=False, init_values=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2, num_experts=4, top_k=2, use_moe=False):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate, act_layer=act_layer, norm_layer=norm_layer, block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group, num_experts=num_experts, top_k=top_k, use_moe=use_moe))

    def forward(self, x, pos):
        feature_lst = []
        idx = knn_point(self.k, pos, pos)
        for (_, block) in enumerate(self.blocks):
            x = block(x, pos, idx=idx)
            feature_lst.append(x)
        return feature_lst

class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """

    def __init__(self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4.0, qkv_bias=True, init_values=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, act_layer=None, block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2, num_experts=4, top_k=2, use_moe=False):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, depth=depth, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=dpr, norm_layer=norm_layer, act_layer=act_layer, block_style_list=block_style_list, combine_style=combine_style, k=k, n_group=n_group, num_experts=num_experts, top_k=top_k, use_moe=use_moe)
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        feature_lst = self.blocks(x, pos)
        return feature_lst

class PointTransformerEncoderEntry(PointTransformerEncoder):

    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class DGCNN_Grouper(nn.Module):

    def __init__(self, k=16):
        super().__init__()
        self.k = k
        self.input_trans = nn.Sequential(nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.ReLU(), nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Conv2d(262, 32, kernel_size=1, bias=False), nn.GroupNorm(4, 32), nn.LeakyReLU(negative_slope=0.2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), nn.GroupNorm(4, 64), nn.LeakyReLU(negative_slope=0.2))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), nn.GroupNorm(4, 64), nn.LeakyReLU(negative_slope=0.2))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.GroupNorm(4, 128), nn.LeakyReLU(negative_slope=0.2))
        self.num_features = 128

    @staticmethod
    def fps_downsample(coor, x, num_group):
        """Downsample points with farthest point sampling."""
        xyz = coor.transpose(1, 2).contiguous()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
        combined_x = torch.cat([coor, x], dim=1)
        new_combined_x = pointnet2_utils.gather_operation(combined_x, fps_idx)
        new_coor = new_combined_x[:, :3, :]
        new_x = new_combined_x[:, 3:, :]
        return (new_coor, new_x)

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        """Build graph features for query points from neighboring key points."""
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)
        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous())
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims)
        feature = feature.permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def _forward(self, x, num):
        """Encode sampled point features at the requested group sizes."""
        x = x.transpose(-1, -2).contiguous()
        coor = x[:, :3, :]
        features = x[:, 3:, :]
        f = self.input_trans(features)
        f = torch.cat([f, coor], dim=1)
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]
        (coor_q, f_q) = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]
        (coor_q, f_q) = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q
        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()
        return (coor, f)

    def forward(self, x, num):
        """Encode sampled point features at the requested group sizes."""
        x = x.transpose(-1, -2).contiguous()
        coor = x[:, :3, :]
        features = x[:, 3:, :]
        f = self.input_trans(features)
        f = torch.cat([f, coor], dim=1)
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]
        (coor_q, f_q) = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]
        (coor_q, f_q) = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q
        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()
        return (coor, f)

class Encoder(nn.Module):

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, self.encoder_channel, 1))

    def forward(self, point_groups):
        """
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        """
        (bs, g, n, _) = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)

class SimpleEncoder(nn.Module):

    def __init__(self, k=32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k
        self.num_features = embed_dims

    def forward(self, xyz, n_group):
        if isinstance(n_group, list):
            n_group = n_group[-1]
        center = misc.fps(xyz, n_group)
        assert center.size(1) == n_group, f'expect center to be B {n_group} 3, but got shape {center.shape}'
        (batch_size, num_points, _) = xyz.shape
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()
        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size
        features = self.embedding(neighborhood)
        return (center, features)

class SimpleRebuildFCLayer(nn.Module):

    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        """
        Input BNC
        """
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
        patch_feature = torch.cat([g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1), token_feature], dim=-1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step, 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc

class SimpleFeatureFusion(nn.Module):

    def __init__(self, num_layers=8):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, feature_lst):
        weights = torch.softmax(self.weights, dim=0)
        fused_feature = sum((w * f for (w, f) in zip(weights, feature_lst)))
        return fused_feature

class SpatialFeatureFusion(nn.Module):
    """
    Fuse a list of spatial feature maps with identical spatial size.

    Args
    ----
    in_channels : int
        #Channels of each incoming feature map (RADIO v2.5 returns 1024).
    mid_channels : int
        Compressed channels before fusion.
    num_levels : int
        Number of feature maps in the list (len(spatial_features_list)).
    out_channels : int
        Channels of the fused output.
    """

    def __init__(self, in_channels: int=1024, mid_channels: int=512, num_levels: int=3, out_channels: int=1024):
        super().__init__()
        self.compress = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True)) for _ in range(num_levels)])
        self.level_weights = nn.Parameter(torch.ones(num_levels))
        self.refine = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        feats : list[Tensor]
            Each tensor has shape (B, C_in, H, W) and the same spatial size.

        Returns
        -------
        fused : Tensor
            Fused features with shape (B, out_channels, H, W).
        """
        assert len(feats) == len(self.compress), f'Expect {len(self.compress)} feature maps, got {len(feats)}.'
        compressed = [conv(f) for (conv, f) in zip(self.compress, feats)]
        weights = F.softmax(self.level_weights, dim=0)
        fused = torch.zeros_like(compressed[0])
        for (w, feat) in zip(weights, compressed):
            fused = fused + w * feat
        fused = self.refine(fused)
        return fused

class RGBDEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        encoder_config = getattr(config, 'encoder_config', None)
        self.encoder_config = encoder_config
        decoder_config = getattr(config, 'decoder_config', None)
        self.decoder_config = decoder_config
        if self.decoder_config is None:

            class DefaultDecoderConfig:

                def __init__(self):
                    self.wo_selection = False
                    self.wo_c2f = False
            self.decoder_config = DefaultDecoderConfig()
        self.center_num = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'
        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)
        self.pos_embed = nn.Sequential(nn.Linear(in_chans, 128), nn.GELU(), nn.Linear(128, encoder_config.embed_dim))
        self.input_proj = nn.Sequential(nn.Linear(self.grouper.num_features, 512), nn.GELU(), nn.Linear(512, encoder_config.embed_dim))
        self.encoder = PointTransformerEncoderEntry(encoder_config)
        self.increase_dim = nn.Sequential(nn.Linear(encoder_config.embed_dim, 1024), nn.GELU(), nn.Linear(1024, global_feature_dim))
        self.coarse_pred = nn.Sequential(nn.Linear(global_feature_dim, 1024), nn.GELU(), nn.Linear(1024, 3 * query_num))
        self.query_ranking = nn.Sequential(nn.Linear(3, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid())
        self.feature_fusion = SimpleFeatureFusion(num_layers=8)
        self.radio_fusion_net = SpatialFeatureFusion(in_channels=1024, mid_channels=512, num_levels=3, out_channels=1024)
        self.apply(self._init_weights)
        self.radio_encoder = load_frozen_radio(config)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz, roi_rgb, roi_rows, roi_columns, return_shape=True):
        self.radio_encoder.eval()
        with torch.no_grad():
            (_, spatial_features_list) = self.radio_encoder.forward_intermediates(roi_rgb, indices=[8, 16, 23])
        fused_feat = self.radio_fusion_net(spatial_features_list)
        (B, C, H, W) = fused_feat.shape
        spatial_features = fused_feat.view(B, C, H * W)
        spatial_features = spatial_features.permute(0, 2, 1)
        feature_rows = roi_rows // 16
        feature_columns = roi_columns // 16
        pos = feature_rows * (roi_rgb.shape[-1] // 16) + feature_columns
        pos = torch.unsqueeze(pos, -1).expand(-1, -1, 1024)
        rgb_feat = torch.gather(spatial_features, 1, pos)
        rgb_feat = rgb_feat.to(xyz.dtype)
        xyzrgb = torch.cat([xyz, rgb_feat], dim=-1)
        bs = xyzrgb.size(0)
        (coor, f) = self.grouper(xyzrgb, self.center_num)
        pe = self.pos_embed(coor)
        x = self.input_proj(f)
        feature_lst = self.encoder(x + pe, coor)
        x = self.feature_fusion(feature_lst)
        global_feature = self.increase_dim(x)
        global_feature = torch.max(global_feature, dim=1)[0]
        if not return_shape:
            return None, global_feature
        if hasattr(self.decoder_config, 'wo_selection') and self.decoder_config.wo_selection:
            coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)
        elif hasattr(self.decoder_config, 'wo_c2f') and self.decoder_config.wo_c2f:
            coarse = None
        else:
            coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)
            coarse_inp = misc.fps(xyz, self.num_query // 2)
            coarse = torch.cat([coarse, coarse_inp], dim=1)
            query_ranking = self.query_ranking(coarse)
            idx = torch.argsort(query_ranking, dim=1, descending=True)
            coarse = torch.gather(coarse, 1, idx[:, :self.num_query].expand(-1, -1, coarse.size(-1)))
        return (coarse, global_feature)

class F2GPose(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.encoder_config.embed_dim
        self.num_query = config.num_query
        self.decoder_config = getattr(config, 'decoder_config', None)
        self.use_aux_loss = getattr(config, 'use_aux_loss', True)
        self.aux_loss_weight = getattr(config, 'aux_loss_weight', 0.01)
        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'
        self.base_model = RGBDEncoder(config)
        self.build_loss_func()
        self.rotat_head = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(256, 6))
        self.trans_head = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(256, 3))
        self.size_head = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(256, 3))
        if self.decoder_type == 'fc':
            self.fold_step = 2
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step ** 2)
            self.reduce_map = nn.Linear(1027, self.trans_dim)
        if self.decoder_config is not None and hasattr(self.decoder_config, 'wo_c2f') and self.decoder_config.wo_c2f:
            self.dense_pred = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 3 * 2048))

    def _generate_rotation_matrix(self, axis, angle_degrees):
        """Generate a rotation matrix for a given axis and angle."""
        angle = torch.tensor(angle_degrees * np.pi / 180)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        if axis == 'x':
            R = torch.tensor([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        elif axis == 'y':
            R = torch.tensor([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        elif axis == 'z':
            R = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        else:
            raise ValueError(f'Unknown axis: {axis}')
        return R

    def _rotate_loss_with_symmetry(self, ortho6d, symmetry, loss_fn, gt_ortho6d, return_per_sample=False):
        """Return the minimum rotation loss over each object's symmetries."""
        if len(ortho6d.shape) == 3:
            (bs, num_hypo, _) = ortho6d.shape
            predicted_rotation_matrices = compute_rotation_matrix_from_ortho6d(ortho6d.view(bs * num_hypo, -1)).view(bs, num_hypo, 3, 3)
            gt_rotation_matrices = compute_rotation_matrix_from_ortho6d(gt_ortho6d.view(bs * num_hypo, -1)).view(bs, num_hypo, 3, 3)
        else:
            predicted_rotation_matrices = compute_rotation_matrix_from_ortho6d(ortho6d)
            gt_rotation_matrices = compute_rotation_matrix_from_ortho6d(gt_ortho6d)
        (nx, ny, nz) = (symmetry['x'], symmetry['y'], symmetry['z'])
        batch_size = ortho6d.shape[0]
        angles_dict = {0: [0], 1: [i for i in range(0, 360)], 3: [0, 90, 180, 270], 2: [0, 180]}
        symmetry_rotations = {'x': {}, 'y': {}, 'z': {}}
        for axis in ['x', 'y', 'z']:
            for sym_param in [0, 1, 2, 3]:
                angles = angles_dict[sym_param]
                rotations = torch.stack([self._generate_rotation_matrix(axis, angle).to(ortho6d.device) for angle in angles], dim=0)
                symmetry_rotations[axis][sym_param] = rotations
        groups = {}
        for i in range(batch_size):
            sym_params = (nx[i].item(), ny[i].item(), nz[i].item())
            if sym_params.count(1) >= 2:
                continue
            if sym_params not in groups:
                groups[sym_params] = []
            groups[sym_params].append(i)
        (all_min_losses, all_min_geo_losses) = ([], [])
        for (sym_params, indices) in groups.items():
            (nx_i, ny_i, nz_i) = sym_params
            rotations_x = symmetry_rotations['x'][nx_i]
            rotations_y = symmetry_rotations['y'][ny_i]
            rotations_z = symmetry_rotations['z'][nz_i]
            rotations_x = rotations_x[:, None, None, :, :]
            rotations_y = rotations_y[None, :, None, :, :]
            rotations_z = rotations_z[None, None, :, :, :]
            R_sym = torch.matmul(torch.matmul(rotations_z, rotations_y), rotations_x)
            R_sym = R_sym.reshape(-1, 3, 3)
            gt_rotation_samples = gt_rotation_matrices[indices]
            gt_rotation_samples = gt_rotation_samples[:, None, :, :]
            R_sym = R_sym[None, :, :, :]
            rotated_gt_matrices = torch.matmul(gt_rotation_samples, R_sym)
            rotated_gt_matrices_flat = rotated_gt_matrices.reshape(-1, 3, 3)
            rotated_gt_6d_flat = torch.cat([rotated_gt_matrices_flat[:, :, 0], rotated_gt_matrices_flat[:, :, 1]], dim=1)
            num_samples_in_group = len(indices)
            num_combinations = R_sym.shape[1]
            rotated_gt_6d = rotated_gt_6d_flat.reshape(num_samples_in_group, num_combinations, 6)
            ortho6d_samples = ortho6d[indices]
            ortho6d_samples_expanded = ortho6d_samples[:, None, :].expand(-1, num_combinations, -1)
            losses = loss_fn(ortho6d_samples_expanded, rotated_gt_6d)
            losses = losses.mean(dim=2)
            (min_loss_vals, _) = torch.min(losses, dim=1)
            all_min_losses.append(min_loss_vals)
        if all_min_losses:
            rotate_loss = torch.cat(all_min_losses, dim=0).mean()
        else:
            rotate_loss = torch.tensor(0.0, device=ortho6d.device)
        return rotate_loss

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def _collect_moe_aux_loss(self):
        """Average the load-balancing loss from MoE layers."""
        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for module in self.modules():
            if isinstance(module, MoeMlp):
                aux_loss = aux_loss + module.aux_loss.to(aux_loss.device)
                count += 1
        if count > 0:
            aux_loss = aux_loss / count
        return aux_loss

    def get_loss(self, ret, gt, gt_rotation_6d, gt_trans_mat, gt_size_mat, gt_symmetry, epoch=1, step=1):
        (pred_coarse, pred_dense, pred_rotation_6d, pred_trans, pred_size) = ret
        loss_dense = self.loss_func(pred_dense.float(), gt.float())
        if pred_coarse is not None:
            loss_coarse = self.loss_func(pred_coarse.float(), gt.float())
        else:
            loss_coarse = torch.tensor(0.0, device=pred_dense.device)
        loss_rotation = self._rotate_loss_with_symmetry(pred_rotation_6d, gt_symmetry, nn.SmoothL1Loss(reduction='none'), gt_rotation_6d)
        loss_trans = nn.SmoothL1Loss()(pred_trans, gt_trans_mat)
        loss_size = nn.SmoothL1Loss()(pred_size, gt_size_mat)
        if self.use_aux_loss:
            loss_aux = self.aux_loss_weight * self._collect_moe_aux_loss()
        else:
            loss_aux = torch.tensor(0.0, device=pred_dense.device)
        return (loss_coarse, loss_dense, loss_rotation, loss_trans, loss_size, loss_aux)

    def forward(self, xyz, roi_rgb, roi_rows, roi_columns, return_shape=True):
        (coarse_point_cloud, global_feature) = self.base_model(xyz, roi_rgb, roi_rows, roi_columns, return_shape=return_shape)
        if not return_shape:
            return (None, None, self.rotat_head(global_feature), self.trans_head(global_feature), self.size_head(global_feature))
        if coarse_point_cloud is not None:
            (B, M, _) = coarse_point_cloud.shape
            rebuild_feature = torch.cat([coarse_point_cloud, global_feature.unsqueeze(-2).expand(-1, M, -1)], dim=-1)
            rebuild_feature = self.reduce_map(rebuild_feature)
            relative_xyz = self.decode_head(rebuild_feature)
            rebuild_points = relative_xyz + coarse_point_cloud.unsqueeze(-2)
            dense_point_cloud = rebuild_points.reshape(B, -1, 3).contiguous()
        else:
            B = global_feature.shape[0]
            dense_point_cloud = self.dense_pred(global_feature).reshape(B, -1, 3)
        rotation_6d = self.rotat_head(global_feature)
        trans = self.trans_head(global_feature)
        size = self.size_head(global_feature)
        ret = (coarse_point_cloud, dense_point_cloud, rotation_6d, trans, size)
        return ret
