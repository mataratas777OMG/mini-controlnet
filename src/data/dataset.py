import os

import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from .utils import compute_canny


class EdgeImageDataset(Dataset):
    def __init__(self, root_dir, size=256, ext=("jpg", "png", "jpeg")):
        self.files = []
        for e in ext:
            self.files += glob(os.path.join(root_dir, f"**/*.{e}"), recursive=True)
        self.files = [f for f in self.files if os.path.isfile(f)]

        assert len(self.files) > 0, f"No images found in {root_dir}!"
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        low = F.interpolate(image.unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False)
        coarse = F.interpolate(low, size=(self.size, self.size), mode='bilinear', align_corners=False).squeeze(0)

        image_np = (np.array(image.resize((self.size, self.size)))[:, :, :3]).astype(np.uint8)
        edges = compute_canny(coarse)
        edges = torch.from_numpy(edges).unsqueeze(0)

        target = image * 2.0 - 1.0
        coarse = coarse * 2.0 - 1.0
        return coarse, edges, target