import os
from PIL import Image
import torch
from torchvision import transforms

clean_root = "Clean_dataset"
noisy_root = "Noisy_dataset"
pre_root = "Pre_dataset"

noise_types = ["gaussian", "salt_pepper", "speckle", "poisson", "uniform"]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

for noise in noise_types:
    print(f"Creating tensor pairs for noise: {noise}")
    noisy_base = os.path.join(noisy_root, noise)
    clean_base = clean_root
    pre_noisy_dir = os.path.join(pre_root, noise, "noisy_tensor")
    pre_clean_dir = os.path.join(pre_root, noise, "clean_tensor")
    os.makedirs(pre_noisy_dir, exist_ok=True)
    os.makedirs(pre_clean_dir, exist_ok=True)

    for root, _, files in os.walk(noisy_base):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            noisy_path = os.path.join(root, file)
            rel_path = os.path.relpath(noisy_path, noisy_base)
            clean_path = os.path.join(clean_base, rel_path)

            if not os.path.isfile(clean_path):
                print(f"⚠️ Missing clean image for {rel_path}, skipping")
                continue

            # Load images
            noisy_img = Image.open(noisy_path).convert("RGB")
            clean_img = Image.open(clean_path).convert("RGB")

            # Transform to tensor
            noisy_tensor = transform(noisy_img)
            clean_tensor = transform(clean_img)

            # Save tensors
            noisy_tensor_path = os.path.join(pre_noisy_dir, rel_path) + ".pt"
            clean_tensor_path = os.path.join(pre_clean_dir, rel_path) + ".pt"

            os.makedirs(os.path.dirname(noisy_tensor_path), exist_ok=True)
            os.makedirs(os.path.dirname(clean_tensor_path), exist_ok=True)

            torch.save(noisy_tensor, noisy_tensor_path)
            torch.save(clean_tensor, clean_tensor_path)

print("✅ Tensor pair creation done.")
