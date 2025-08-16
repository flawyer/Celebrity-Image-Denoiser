import os
from PIL import Image
import numpy as np
import cv2

# Define noise functions
def add_gaussian(img):
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

def add_salt_pepper(img, amount=0.004):
    out = np.copy(img)
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i, int(num_salt))
              for i in img.shape]
    out[tuple(coords)] = 1
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i, int(num_pepper))
              for i in img.shape]
    out[tuple(coords)] = 0
    return out

def add_speckle(img):
    noise = np.random.randn(*img.shape)
    noisy = img + img * noise
    return np.clip(noisy, 0, 1)

def add_poisson(img):
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img * vals) / float(vals)
    return np.clip(noisy, 0, 1)

def add_uniform(img):
    noise = np.random.uniform(-0.05, 0.05, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

# Paths
clean_root = "Clean_dataset"
noisy_root = "Noisy_dataset"
noise_types = ["gaussian", "salt_pepper", "speckle", "poisson", "uniform"]

for noise_type in noise_types:
    print(f"Generating {noise_type} noise...")
    for root, _, files in os.walk(clean_root):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            clean_path = os.path.join(root, file)
            rel_path = os.path.relpath(clean_path, clean_root)
            noisy_subfolder = os.path.join(noisy_root, noise_type, os.path.dirname(rel_path))
            os.makedirs(noisy_subfolder, exist_ok=True)

            # Load + normalize
            img = np.array(Image.open(clean_path).convert("RGB")) / 255.0

            if noise_type == "gaussian":
                noisy_img = add_gaussian(img)
            elif noise_type == "salt_pepper":
                noisy_img = add_salt_pepper(img)
            elif noise_type == "speckle":
                noisy_img = add_speckle(img)
            elif noise_type == "poisson":
                noisy_img = add_poisson(img)
            elif noise_type == "uniform":
                noisy_img = add_uniform(img)

            noisy_img = (noisy_img * 255).astype(np.uint8)
            Image.fromarray(noisy_img).save(os.path.join(noisy_subfolder, file))

print("✅ Noise generation complete. Check Noisy_dataset/")
