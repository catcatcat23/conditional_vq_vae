import torch
import torch.nn as nn
import torch.nn.functional as F
from .panns_encoder import PANNSEncoder
from .vector_quantizer import VectorQuantizer   
from .decoder import UTransformerDecoder

class ConditonalVQVAE(nn.Module):
    def __init__(self, num_labels, feature_dim,latent_dim = 64,codebook_size=512, label_embed_dim=16):

        super().__init__()
        self.label_embed = nn.Embedding(num_labels, label_embed_dim)

        # encoder 输出 skips
        self.encoder = PANNSEncoder(latent_dim, label_embed_dim,return_skips=True)

        self.quantizer = VectorQuantizer(codebook_size, latent_dim, commitment_cost=0.25)  

        #使用sota transformer decoder
        self.decoder = UTransformerDecoder(latent_dim, label_embed_dim, feature_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, num_labels)
        )        
        self.last_indices = None



    def forward(self, x, y):
        """
        x: [B, T_in, feature_dim]
        y: [B]
        """
        y_emb = self.label_embed(y)                  # [B, L]
        z_e, skips = self.encoder(x, y_emb)          # z_e: [B, D, T/32]
        z_q, vq_loss, indices = self.quantizer(z_e)           # 量化

        # 这里传入 target_length = x.size(1)
        T_in    = x.size(1)
        x_recon = self.decoder(z_q, y_emb, skips, target_length=T_in)

        self.last_indices = indices  # 形状 [B, T_lat]

        logits  = self.classifier(z_q)               # 分类
        return x_recon, logits, vq_loss