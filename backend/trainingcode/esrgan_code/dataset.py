import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class NoisyCleanPairDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, size=256):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

        self.pairs = []
        for root, _, files in os.walk(noisy_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    noisy_path = os.path.join(root, f)
                    rel_path = os.path.relpath(noisy_path, noisy_dir)
                    clean_path = os.path.join(clean_dir, rel_path)
                    if os.path.isfile(clean_path):
                        self.pairs.append((noisy_path, clean_path))
                    else:
                        print(f"⚠️ No matching clean file for: {rel_path}")

        # Transform: Resize + Tensor
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy = Image.open(noisy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")
        return self.transform(noisy), self.transform(clean)
