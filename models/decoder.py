# models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

class UpBlock(nn.Module):
    """U‑Net 风格上采样 + 跳跃连接"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        """
        x:    [B, in_ch,  T  ]
        skip: [B, skip_ch, 2T ]
        """
        x = self.up(x)                   # → [B, out_ch, 2T]
        x = torch.cat([x, skip], dim=1)  # → [B, out_ch+skip_ch, 2T]
        return self.conv(x)              # → [B, out_ch, 2T]


class UpBlockNoSkip(nn.Module):
    """最后一级只上采样，不拼接跳连"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: [B, in_ch, T]
        """
        x = self.up(x)   # → [B, out_ch, 2T]
        return self.conv(x)  # → [B, out_ch, 2T]


class UTransformerDecoder(nn.Module):
    """
    精准逆转 5 次下采样：T/32 → T/16 → T/8 → T/4 → T/2 → T
    最后一层不拼接 skip，只上采样。
    """
    def __init__(self, latent_dim, label_embed_dim, feature_dim):
        super().__init__()
        # 标签映射
        self.label_proj = nn.Linear(label_embed_dim, latent_dim)
        # 自注意力瓶颈
        self.attn = TransformerEncoderLayer(
            d_model=latent_dim, nhead=4, dim_feedforward=latent_dim*4
        )

        # 上采样模块：
        # skip_ch 的通道数需与 Encoder 返回的 skip 对应：
        #   skips_for_decoder = [skip4@512, skip3@256, skip2@128, skip1@64, skip1@64]
        self.up5 = UpBlock(      latent_dim, skip_ch=512, out_ch=512 )  # T/32→T/16
        self.up4 = UpBlock(     512,      skip_ch=256, out_ch=256 )    # T/16→T/8
        self.up3 = UpBlock(     256,      skip_ch=128, out_ch=128 )    # T/8 →T/4
        self.up2 = UpBlock(     128,      skip_ch=64,  out_ch=64  )    # T/4 →T/2
        self.up1 = UpBlockNoSkip(64,      out_ch=32 )                  # T/2 →T

        # 最后 1×1 卷积恢复原始 feature_dim
        self.final_conv = nn.Conv1d(32, feature_dim, kernel_size=1)

    def forward(self, z_q, label_emb, skips, target_length: int = None):
        B, D, Tp = z_q.shape
        # 标签融合
        lm = self.label_proj(label_emb).unsqueeze(2).expand(-1,-1,Tp)
        h  = z_q + lm
        # 自注意力瓶颈
        h = self.attn(h.permute(2,0,1)).permute(1,2,0)
        # 5 级上采样
        h = self.up5(h, skips[0])
        h = self.up4(h, skips[1])
        h = self.up3(h, skips[2])
        h = self.up2(h, skips[3])
        h = self.up1(h)               # 现在 h.shape == [B,32,T_recon]，T_recon = floor(T_in/32)*32
        # 恢复 feature_dim
        out = self.final_conv(h)      # [B, feature_dim, T_recon]

        # —— 末尾插值到 target_length —— #
        if target_length is not None and out.size(2) != target_length:
            out = F.interpolate(
                out,
                size=target_length,
                mode='linear',
                align_corners=False
            )
        # 返回 [B, T, feature_dim]
        return out.permute(0,2,1)