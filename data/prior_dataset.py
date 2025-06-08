import torch
from torch.utils.data import Dataset

class PriorDataset(Dataset):
    """
    接收两个 .pt 文件：
      - indices_path: [N, T_lat] 的离散索引张量
      - labels_path:  [N] 的类别标签张量
    返回 (索引序列, 标签) 对：
      - 索引序列：LongTensor, 形状 [T_lat]
      - 标签：LongTensor, 标量
    """
    def __init__(self, indices_path, labels_path):
        # 1) 把预先保存好的索引和标签 load 进来
        #    all_indices: [N, T_lat]
        #    all_labels:  [N]
        self.all_indices = torch.load(indices_path)  # LongTensor
        self.all_labels  = torch.load(labels_path)   # LongTensor

        assert self.all_indices.ndim == 2
        assert self.all_labels.ndim == 1
        assert self.all_indices.size(0) == self.all_labels.size(0)

    def __len__(self):
        return self.all_indices.size(0)

    def __getitem__(self, idx):
        # 返回：
        #   seq: 形状 [T_lat], dtype=torch.long
        #   label: 单个整数 torch.long
        seq = self.all_indices[idx]      # LongTensor, [T_lat]
        lbl = self.all_labels[idx]       # LongTensor, scalar
        return seq, lbl
