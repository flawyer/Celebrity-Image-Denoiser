import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import lpips
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# === Models ===
from models import Generator, Discriminator

# === Dataset ===
class TensorPairDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.noisy_files = []
        for root, _, files in os.walk(self.noisy_dir):
            for file in files:
                if file.endswith('.pt'):
                    rel_path = os.path.relpath(os.path.join(root, file), self.noisy_dir)
                    self.noisy_files.append(rel_path)
        self.noisy_files.sort()

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_tensor = torch.load(os.path.join(self.noisy_dir, self.noisy_files[idx]))
        clean_tensor = torch.load(os.path.join(self.clean_dir, self.noisy_files[idx]))
        return noisy_tensor, clean_tensor

# === Folders ===
os.makedirs("checkpoints/step2", exist_ok=True)
os.makedirs("checkpoints/best", exist_ok=True)
os.makedirs("Graphs", exist_ok=True)
os.makedirs("ESR_TestImg", exist_ok=True)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hyperparameters ===
batch_size = 8
lr = 1e-4
target_noise = "gaussian"

# === Dataset ===
dataset = TensorPairDataset(
    noisy_dir=f"Pre_dataset/{target_noise}/noisy_tensor",
    clean_dir=f"Pre_dataset/{target_noise}/clean_tensor"
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# === Models ===
G = Generator().to(device)
D = Discriminator().to(device)

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

g_opt = torch.optim.Adam(G.parameters(), lr=lr)
d_opt = torch.optim.Adam(D.parameters(), lr=lr)

# === Save test image ===
def save_test_image(epoch, noisy, gen, clean):
    noisy_np = noisy.cpu().permute(1,2,0).numpy().clip(0,1)
    gen_np = gen.cpu().permute(1,2,0).numpy().clip(0,1)
    clean_np = clean.cpu().permute(1,2,0).numpy().clip(0,1)
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(noisy_np); ax[0].set_title("Noisy"); ax[0].axis('off')
    ax[1].imshow(gen_np); ax[1].set_title("Generated"); ax[1].axis('off')
    ax[2].imshow(clean_np); ax[2].set_title("Clean"); ax[2].axis('off')
    plt.tight_layout()
    plt.savefig(f"ESR_TestImg/epoch_{epoch}.png")
    plt.close()

# === Train ===
num_epochs = int(input("Enter number of epochs to train: "))
best_psnr = 0

D_losses, G_losses, PSNRs, SSIMs, LPIPSs, MSSIMs = [], [], [], [], [], []

for epoch in range(num_epochs):
    G.train()
    D.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    epoch_psnr, epoch_ssim, epoch_lpips, epoch_mssim = 0,0,0,0
    count = 0

    for noisy, clean in pbar:
        noisy = noisy.to(device)
        clean = clean.to(device)

        # --- Train D ---
        denoised = G(noisy)
        d_opt.zero_grad()
        real_out = D(clean)
        fake_out = D(denoised.detach())

        real_label = torch.ones_like(real_out)
        fake_label = torch.zeros_like(fake_out)

        d_loss = 0.5 * (bce_loss(real_out, real_label) + bce_loss(fake_out, fake_label))
        d_loss.backward()
        d_opt.step()

        # --- Train G ---
        g_opt.zero_grad()
        fake_out_g = D(denoised)
        gan_loss = bce_loss(fake_out_g, real_label)
        pixel_loss = mse_loss(denoised, clean)
        g_loss = pixel_loss + 1e-3 * gan_loss
        g_loss.backward()
        g_opt.step()

        # --- Metrics ---
        with torch.no_grad():
            den_clamp = denoised.clamp(0,1)
            for i in range(den_clamp.size(0)):
                psnr = compare_psnr(clean[i].cpu().numpy().transpose(1,2,0),
                                    den_clamp[i].cpu().numpy().transpose(1,2,0),
                                    data_range=1.0)
                ssim = compare_ssim(clean[i].cpu().numpy().transpose(1,2,0),
                                    den_clamp[i].cpu().numpy().transpose(1,2,0),
                                    channel_axis=2, data_range=1.0)
                lpips_val = lpips_loss_fn(den_clamp[i].unsqueeze(0), clean[i].unsqueeze(0)).item()
                epoch_psnr += psnr
                epoch_ssim += ssim
                epoch_lpips += lpips_val
                count += 1

        pbar.set_postfix(D=d_loss.item(), G=g_loss.item(), PSNR=epoch_psnr/count)

    # === Save metrics ===
    avg_psnr = epoch_psnr / count
    avg_ssim = epoch_ssim / count
    avg_lpips = epoch_lpips / count
    MSSIMs.append(avg_ssim)
    PSNRs.append(avg_psnr)
    SSIMs.append(avg_ssim)
    LPIPSs.append(avg_lpips)
    D_losses.append(d_loss.item())
    G_losses.append(g_loss.item())

    print(f"Epoch {epoch+1}: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f}")

    # === Save test ===
    G.eval()
    with torch.no_grad():
        idx = random.randint(0, len(dataset)-1)
        noisy_t, clean_t = dataset[idx]
        noisy_t = noisy_t.to(device).unsqueeze(0)
        clean_t = clean_t.to(device).unsqueeze(0)
        gen_t = G(noisy_t).clamp(0,1)
        save_test_image(epoch+1, noisy_t[0], gen_t[0], clean_t[0])

    # === Save checkpoints ===
    if epoch+1 == 1 or (epoch+1) == num_epochs or (epoch+1) % 2 == 0:
        ckpt = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'g_opt': g_opt.state_dict(),
            'd_opt': d_opt.state_dict(),
            'epoch': epoch+1
        }
        torch.save(ckpt, f"checkpoints/step2/epoch_{epoch+1}.pth")
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(ckpt, f"checkpoints/best/best_epoch_{epoch+1}.pth")

# === Plot individual graphs ===
plt.plot(D_losses); plt.title("Discriminator Loss"); plt.savefig("Graphs/discriminator_loss.png"); plt.close()
plt.plot(G_losses); plt.title("Generator Loss"); plt.savefig("Graphs/generator_loss.png"); plt.close()
plt.plot(PSNRs); plt.title("PSNR"); plt.savefig("Graphs/psnr.png"); plt.close()
plt.plot(SSIMs); plt.title("SSIM"); plt.savefig("Graphs/ssim.png"); plt.close()
plt.plot(LPIPSs); plt.title("LPIPS"); plt.savefig("Graphs/lpips.png"); plt.close()
plt.plot(MSSIMs); plt.title("MS-SSIM"); plt.savefig("Graphs/ms-ssim.png"); plt.close()

# === Combined final summary ===
fig, axes = plt.subplots(2, 3, figsize=(18,10))
axes[0,0].plot(D_losses); axes[0,0].set_title("Discriminator Loss")
axes[0,1].plot(G_losses); axes[0,1].set_title("Generator Loss")
axes[0,2].plot(PSNRs); axes[0,2].set_title("PSNR")
axes[1,0].plot(SSIMs); axes[1,0].set_title("SSIM")
axes[1,1].plot(LPIPSs); axes[1,1].set_title("LPIPS")
axes[1,2].plot(MSSIMs); axes[1,2].set_title("MS-SSIM")
plt.tight_layout()
plt.savefig("Graphs/training_summary.png")
plt.close()

print("✅ Training done. All graphs & checkpoints saved.")
