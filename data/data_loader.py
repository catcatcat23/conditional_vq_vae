import os
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_MAP = {
    "正常":   0,
    "湿罗音": 1,
    "都有": 2,
}

class RespiratoryDataset(Dataset):
    def __init__(self, npy_dir, label_dir,transform=None):
        super().__init__()
        self.npy_dir = npy_dir
        self.label_dir = label_dir
        self.transform = transform
        self.file_name = sorted(fn for fn in os.listdir(npy_dir) if fn.endswith('.npy'))

    def __len__(self):
        return len(self.file_name)  
    
    def __getitem__(self, idx):
        fn = self.file_name[idx]

        # 加载
        x = np.load(os.path.join(self.npy_dir, fn)) # [T,F]

        # 读取标签
        label_path = os.path.join(self.label_dir, fn.replace('.npy', '.txt'))
        with open(label_path, 'r', encoding='utf-8') as f:
            label = f.read().strip()
        if label not in LABEL_MAP:  
            raise ValueError(f"Unknown label: {label}")
        
        y = LABEL_MAP[label]

        if self.transform:
            x = self.transform(x)   

        return x,y