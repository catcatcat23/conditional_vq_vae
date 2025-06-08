import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 初始化码本（Embedding 矩阵）
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    def forward(self, x):
        """
        x: [B, D, T_lat] 或 [B, T_lat, D]（你在调用时先确保传入的是 [B, T_lat, D]）
        """
        # 如果输入是 [B, D, T_lat]，先把它转成 [B, T_lat, D]
        if x.dim() == 3 and x.shape[1] == self.embedding_dim:
            # 假设 x.shape == [B, D, T_lat]
            x = x.permute(0, 2, 1).contiguous()  # 变成 [B, T_lat, D]

        B, T_lat, D = x.shape
        # 1) flatten 到 [B*T_lat, D]
        x_flattened = x.view(-1, self.embedding_dim)  # [B*T_lat, D]

        # 2) 计算与所有码本向量的距离： [B*T_lat, K]
        #    公式：||x||^2 + ||e||^2 - 2 x·e
        #    (x_flattened**2).sum(dim=1, keepdim=True) -> [B*T_lat, 1]
        #    (self.embeddings.weight**2).sum(dim=1)       -> [K]
        distances = (
            x_flattened.pow(2).sum(dim=1, keepdim=True)               # [B*T_lat, 1]
            + self.embeddings.weight.pow(2).sum(dim=1)                # [K]
            - 2 * torch.matmul(x_flattened, self.embeddings.weight.t())  # [B*T_lat, K]
        )

        # 3) 找最小值：每个行向量都返回最接近的码本索引
        #    indices_flat: [B*T_lat], 每个元素 ∈ [0..K-1]
        _, indices_flat = torch.min(distances, dim=1)

        # 4) 量化：直接用 indices_flat 从码本里取向量，再 reshape 回 [B, T_lat, D]
        quantized_flat = self.embeddings(indices_flat)  # [B*T_lat, D]
        quantized = quantized_flat.view(B, T_lat, D)    # [B, T_lat, D]

        # 5) 计算 VQ 损失
        #    把 quantized.detach() 当作“常量” vs x 的 MSE，以及 vice versa
        loss = F.mse_loss(quantized.detach(), x) \
             + self.commitment_cost * F.mse_loss(quantized, x.detach())

        # 6) 保证梯度从 quantized 反传给 x
        quantized = x + (quantized - x).detach()

        # 7) 把 indices_flat 还原成 [B, T_lat]
        indices = indices_flat.view(B, T_lat)  # LongTensor

        # 如果你要保持原来的输出顺序，把 quantized 再变成 [B, D, T_lat]：
        quantized = quantized.permute(0, 2, 1).contiguous()  # [B, D, T_lat]

        return quantized, loss, indices
