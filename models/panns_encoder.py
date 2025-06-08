import os
import sys
import torch
import torch.nn as nn

# 确保能 import 到 third_party 下的 pytorch 目录
repo_pytorch = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..', 'third_party', 'audioset_tagging_cnn', 'pytorch'
))
if repo_pytorch not in sys.path:
    sys.path.insert(0, repo_pytorch)

from third_party.audioset_tagging_cnn.pytorch.models import Cnn14

class PANNSEncoder(nn.Module):
    def __init__(self, latent_dim, label_embed_dim, return_skips=False):
        super().__init__()
        self.return_skips = return_skips

        # 2) Backbone：Cnn14
        self.panns = Cnn14(
            sample_rate=32000, window_size=1024, hop_size=320,
            mel_bins=163, fmin=50, fmax=14000, classes_num=527
        )
        # 跳过不需要的模块
        self.panns.fc_audioset         = nn.Identity()
        self.panns.spectrogram_extractor = nn.Identity()
        self.panns.logmel_extractor      = nn.Identity()
        self.panns.spec_augmenter        = nn.Identity()
        self.panns.bn0                   = nn.Identity()

        # 3) 加载预训练权重，剔除前端和分类头
        ckpt = torch.load(
            "/root/autodl-tmp/呼吸音分类/VAE/weights/Cnn14_mAP=0.431.pth",
            map_location='cpu'
        )
        sd   = ckpt.get('model', ckpt)
        for k in list(sd.keys()):
            if k.startswith('logmel_extractor.melW') or k.startswith('fc_audioset'):
                sd.pop(k)
        self.panns.load_state_dict(sd, strict=False)

        # 4) 投射 & 标签嵌入映射
        self.proj       = nn.Conv1d(2048, latent_dim, kernel_size=1)
        self.label_proj = nn.Linear(label_embed_dim, latent_dim)

        # 5) 跳连投射层 对应 conv_block1–5 输出通道 [64,128,256,512,1024]
        self.skip1 = nn.Conv1d(  64,   64, kernel_size=1)  # conv_block1 →64
        self.skip2 = nn.Conv1d( 128,  128, kernel_size=1)  # conv_block2 →128
        self.skip3 = nn.Conv1d( 256,  256, kernel_size=1)  # conv_block3 →256
        self.skip4 = nn.Conv1d( 512,  512, kernel_size=1)  # conv_block4 →512
        self.skip5 = nn.Conv1d(1024, 1024, kernel_size=1)  # conv_block5 →1024

    def forward(self, x, label_emb):
        """
        x:         [B, T, F]
        label_emb: [B, label_embed_dim]
        returns z_e and skips=[skip5..skip1] if return_skips else z_e
        """
        mel = x.unsqueeze(1)  # [B,1,T,F]

        # 5 级下采样（pool2×2）
        x1 = self.panns.conv_block1(mel, pool_size=(2,2), pool_type='avg')  # [B,64, T/2,  F/2]
        x2 = self.panns.conv_block2(x1,  pool_size=(2,2), pool_type='avg')  # [B,128,T/4,  F/4]
        x3 = self.panns.conv_block3(x2,  pool_size=(2,2), pool_type='avg')  # [B,256,T/8,  F/8]
        x4 = self.panns.conv_block4(x3,  pool_size=(2,2), pool_type='avg')  # [B,512,T/16, F/16]
        x5 = self.panns.conv_block5(x4,  pool_size=(2,2), pool_type='avg')  # [B,1024,T/32,F/32]

        # 最终特征（不下采样时间）
        feat_4d = self.panns.conv_block6(x5, pool_size=(1,1), pool_type='avg')  # [B,2048,T/32,F']

        # 跳连：频率维 mean → 3D → 投射
        skip1 = self.skip1(x1.mean(dim=-1))  # [B,  64, T/2 ]
        skip2 = self.skip2(x2.mean(dim=-1))  # [B, 128, T/4 ]
        skip3 = self.skip3(x3.mean(dim=-1))  # [B, 256, T/8 ]
        skip4 = self.skip4(x4.mean(dim=-1))  # [B, 512, T/16]
        skip5 = self.skip5(x5.mean(dim=-1))  # [B,1024, T/32]

        # 投射到 latent + 融合标签
        feat = feat_4d.mean(dim=-1)          # [B,2048,T/32]
        z_e  = self.proj(feat)               # [B, latent_dim, T/32]
        B, D, Tp = z_e.shape
        lm = self.label_proj(label_emb).unsqueeze(2).expand(-1,-1,Tp)  # [B, D, T/32]
        z_e = z_e + lm
        if self.return_skips:
            # raw skips = [skip5@T/32, skip4@T/16, skip3@T/8, skip2@T/4, skip1@T/2]
            # decoder 需要：
            #   up5 用 skip4@T/16
            #   up4 用 skip3@T/8
            #   up3 用 skip2@T/4
            #   up2 用 skip1@T/2
            #   up1 还可以复用 skip1@T/2 （或者 zeros）
            skips_for_decoder = [
                skip4,  # to up5
                skip3,  # to up4
                skip2,  # to up3
                skip1,  # to up2
                skip1   # to up1
            ]
            return z_e, skips_for_decoder
        return z_e