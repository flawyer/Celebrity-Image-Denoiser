import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import pytorch_msssim
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# Denoise Generator (U-Net like structure for denoising)
class DenoiseGenerator(nn.Module):
    def __init__(self):
        super(DenoiseGenerator, self).__init__()
        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.down1(x)
        p1 = self.pool1(e1)
        e2 = self.down2(p1)
        p2 = self.pool2(e2)
        # Bottleneck
        b = self.bottleneck(p2)
        # Decoder with skip connections
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.upconv2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.upconv1(d1)
        return torch.tanh(d1)

# Denoise Discriminator
class DenoiseDiscriminator(nn.Module):
    def __init__(self):
        super(DenoiseDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)

# VGG Perceptual Loss
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.slice.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        return self.criterion(self.slice(x), self.slice(y))

# Custom Dataset (Preprocessing)
class DenoiseDataset(Dataset):
    def __init__(self, noisy_base_dir, clean_dir, noise_types, lr_size=(64, 64), hr_size=(256, 256), test_split=0.2):
        self.noisy_base_dir = noisy_base_dir
        self.clean_dir = clean_dir
        self.noise_types = noise_types
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.image_pairs = []
        self.test_image_pairs = []
        
        # Collect all image pairs and split into train/test
        all_pairs = []
        for noise_type in noise_types:
            noise_dir = os.path.join(noisy_base_dir, noise_type)
            if not os.path.exists(noise_dir):
                print(f"Warning: Noise directory {noise_dir} does not exist.")
                continue
            for person_dir in os.listdir(noise_dir):
                person_noise_dir = os.path.join(noise_dir, person_dir)
                person_clean_dir = os.path.join(clean_dir, person_dir)
                if os.path.isdir(person_noise_dir) and os.path.exists(person_clean_dir):
                    for filename in os.listdir(person_noise_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            noisy_path = os.path.join(person_noise_dir, filename)
                            clean_path = os.path.join(person_clean_dir, filename)
                            if os.path.exists(clean_path):
                                all_pairs.append((noisy_path, clean_path))
        
        # Split into train and test sets
        if all_pairs:
            self.image_pairs, self.test_image_pairs = train_test_split(all_pairs, test_size=test_split, random_state=42)
        else:
            raise ValueError("No valid image pairs found. Check dataset paths and files.")
        
        print(f"Loaded {len(self.image_pairs)} training image pairs and {len(self.test_image_pairs)} test image pairs.")
        if len(self.image_pairs) == 0:
            raise ValueError("No valid training image pairs found. Check dataset paths and files.")
        
        self.noisy_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.clean_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        noisy_path, clean_path = self.image_pairs[idx]
        try:
            noisy_img = Image.open(noisy_path).convert('RGB')
            clean_img = Image.open(clean_path).convert('RGB')
            return self.noisy_transform(noisy_img), self.clean_transform(clean_img)
        except Exception as e:
            print(f"Error loading images: {noisy_path}, {clean_path}. Error: {str(e)}")
            return None

def custom_collate(batch):
    """Filter out None values from the batch."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def plot_metrics(metric_history, graph_dir):
    """Plot all metrics in a single figure with subplots and individual plots using straight lines."""
    os.makedirs(graph_dir, exist_ok=True)
    
    # Check if metric_history has data
    if not metric_history or not any(metric_history.values()):
        print("No metric data available to plot.")
        return
    
    # Create a single figure with subplots (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Results Summary', fontsize=16)
    axes = axes.flatten()
    metrics = ['g_loss', 'd_loss', 'psnr', 'ssim', 'lpips', 'msssim']
    titles = ['Generator Loss', 'Discriminator Loss', 'PSNR', 'SSIM', 'LPIPS', 'MS-SSIM']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        if metric_history[metric]:  # Check if the metric has data
            axes[idx].plot(range(1, len(metric_history[metric]) + 1), metric_history[metric], linestyle='-', marker=None)
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(title)
            axes[idx].grid(True)
        else:
            axes[idx].text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    combined_path = os.path.join(graph_dir, 'training_results.png')
    plt.savefig(combined_path)
    plt.close()
    print(f"Saved combined training results plot: {combined_path}")
    
    # Save individual plots
    for metric, title in zip(metrics, titles):
        if metric_history[metric]:  # Check if the metric has data
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(metric_history[metric]) + 1), metric_history[metric], linestyle='-', marker=None)
            plt.title(f'{title} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(title)
            plt.grid(True)
            individual_path = os.path.join(graph_dir, f'{metric}.png')
            plt.savefig(individual_path)
            plt.close()
            print(f"Saved individual {title} plot: {individual_path}")
        else:
            print(f"No data available for {title} plot.")

# Denoise GAN Trainer
class DenoiseGANTrainer:
    def __init__(self, noisy_base_dir, clean_dir, checkpoint_dir='checkpoint', graph_dir='graphs', test_image_dir='testImage', num_epochs=20, batch_size=16, image_size=(256, 256)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = DenoiseGenerator().to(self.device)
        self.discriminator = DenoiseDiscriminator().to(self.device)
        self.vgg_loss = VGGPerceptualLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)
        self.bce_loss = nn.BCELoss().to(self.device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.scheduler_g = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=30, gamma=0.1)
        self.scheduler_d = optim.lr_scheduler.StepLR(self.d_optimizer, step_size=30, gamma=0.1)
        
        self.dataset = DenoiseDataset(
            noisy_base_dir,
            clean_dir,
            ['gaussian', 'salt_pepper', 'speckle', 'poisson', 'uniform'],
            lr_size=image_size,
            hr_size=image_size,
            test_split=0.2
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate
        )
        
        self.checkpoint_dir = checkpoint_dir
        self.graph_dir = graph_dir
        self.test_image_dir = test_image_dir
        self.best_dir = os.path.join(checkpoint_dir, 'best')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.test_image_dir, exist_ok=True)
        
        self.num_epochs = num_epochs
        self.best_psnr = 0.0
        self.metric_history = {
            'g_loss': [],
            'd_loss': [],
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'msssim': []
        }
        try:
            self.lpips_loss = lpips.LPIPS(net='alex', pretrained=True).to(self.device)
            self.msssim_loss = pytorch_msssim.MS_SSIM(data_range=1.0).to(self.device)
        except Exception as e:
            print(f"Error initializing LPIPS or MS-SSIM: {str(e)}")
            self.lpips_loss = None
            self.msssim_loss = None
        
        # Select one random test image
        self.test_image = self.select_random_test_image()
    
    def select_random_test_image(self):
        """Select one random image from the test set."""
        if not self.dataset.test_image_pairs:
            print("No test images available.")
            return None
        test_image = random.choice(self.dataset.test_image_pairs)
        print(f"Selected test image: {test_image[0]}")
        return test_image
    
    def combine_test_images(self, noisy_img, denoised_img, output_size=(256, 256)):
        """Combine noisy and denoised images side-by-side with labels."""
        noisy_img = noisy_img.resize(output_size, resample=Image.Resampling.BICUBIC)
        denoised_img = denoised_img.resize(output_size, resample=Image.Resampling.BICUBIC)
        
        # Create canvas for two images side-by-side with space for labels
        combined_img = Image.new('RGB', (output_size[0] * 2, output_size[1] + 30), (255, 255, 255))
        combined_img.paste(noisy_img, (0, 30))
        combined_img.paste(denoised_img, (output_size[0], 30))
        
        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((output_size[0]//4, 5), "Noisy", fill=(0, 0, 0), font=font)
        draw.text((output_size[0] + output_size[0]//4, 5), "Denoised", fill=(0, 0, 0), font=font)
        
        return combined_img
    
    def test_random_images(self, epoch, image_size=(256, 256)):
        """Test the selected random image and save results with epoch name."""
        self.generator.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        if not self.test_image:
            print("No test image selected for testing.")
            return
        
        try:
            # Load and preprocess image
            noisy_path, clean_path = self.test_image
            noisy_img = Image.open(noisy_path).convert('RGB')
            img_tensor = transform(noisy_img).unsqueeze(0).to(self.device)
            
            # Generate image
            with torch.no_grad():
                denoised_tensor = self.generator(img_tensor)
            
            # Post-process denoised image
            denoised_img = denoised_tensor.squeeze(0) * 0.5 + 0.5
            denoised_pil = transforms.ToPILImage()(denoised_img.cpu())
            
            # Combine images
            combined_img = self.combine_test_images(noisy_img, denoised_pil)
            
            # Save combined image
            output_path = os.path.join(self.test_image_dir, f'testimg_epoch{epoch}.jpg')
            combined_img.save(output_path)
            print(f"Saved test image: {output_path}")
            
        except Exception as e:
            print(f"Error processing test image {noisy_path} for epoch {epoch}: {str(e)}")
    
    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'epoch': epoch,
            'best_psnr': self.best_psnr,
            'metric_history': self.metric_history
        }
        
        if epoch == 0 or epoch == self.num_epochs - 1 or epoch % 2 == 0 or epoch % 4 == 0:
            torch.save(state, os.path.join(self.checkpoint_dir, f'denoise_epoch_{epoch}.pth'))
        
        if is_best:
            torch.save(state, os.path.join(self.best_dir, f'denoise_epoch_{epoch}.pth'))
    
    def evaluate(self, denoised_hr, clean_hr):
        denoised_np = denoised_hr.detach().cpu().numpy().transpose(0, 2, 3, 1)
        clean_np = clean_hr.detach().cpu().numpy().transpose(0, 2, 3, 1)
        
        psnr_val = np.mean([psnr(denoised_np[i], clean_np[i], data_range=2.0) for i in range(denoised_np.shape[0])])
        ssim_val = np.mean([ssim(denoised_np[i], clean_np[i], channel_axis=2, data_range=2.0) for i in range(denoised_np.shape[0])])
        
        lpips_val = 0.0
        msssim_val = 0.0
        if self.lpips_loss and self.msssim_loss:
            with torch.no_grad():
                lpips_val = self.lpips_loss(denoised_hr * 0.5 + 0.5, clean_hr * 0.5 + 0.5).mean().item()
                msssim_val = self.msssim_loss(denoised_hr * 0.5 + 0.5, clean_hr * 0.5 + 0.5).item()
        
        return psnr_val, ssim_val, lpips_val, msssim_val
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.generator.train()
            self.discriminator.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_metrics = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'msssim': 0.0}
            num_batches = 0
            
            for batch in tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                if batch is None:
                    continue
                noisy_img, clean_img = batch
                noisy_img, clean_img = noisy_img.to(self.device), clean_img.to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                denoised_img = self.generator(noisy_img)
                real_pred = self.discriminator(clean_img)
                fake_pred = self.discriminator(denoised_img.detach())
                d_loss = self.bce_loss(real_pred, torch.ones_like(real_pred)) + \
                         self.bce_loss(fake_pred, torch.zeros_like(fake_pred))
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                fake_pred = self.discriminator(denoised_img)
                content_loss = self.mse_loss(denoised_img, clean_img)
                adv_loss = self.bce_loss(fake_pred, torch.ones_like(fake_pred))
                g_loss = content_loss + 0.001 * adv_loss
                g_loss.backward()
                self.g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
                # Evaluate metrics
                psnr_val, ssim_val, lpips_val, msssim_val = self.evaluate(denoised_img, clean_img)
                epoch_metrics['psnr'] += psnr_val
                epoch_metrics['ssim'] += ssim_val
                epoch_metrics['lpips'] += lpips_val
                epoch_metrics['msssim'] += msssim_val
                num_batches += 1
            
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            if num_batches > 0:
                epoch_g_loss /= num_batches
                epoch_d_loss /= num_batches
                for key in epoch_metrics:
                    epoch_metrics[key] /= num_batches
                
                # Update metric history
                self.metric_history['g_loss'].append(epoch_g_loss)
                self.metric_history['d_loss'].append(epoch_d_loss)
                for key in epoch_metrics:
                    self.metric_history[key].append(epoch_metrics[key])
                
                # Print metrics
                print(f'Epoch [{epoch+1}/{self.num_epochs}] G Loss: {epoch_g_loss:.4f}, D Loss: {epoch_d_loss:.4f}')
                print(f'PSNR: {epoch_metrics["psnr"]:.4f}, SSIM: {epoch_metrics["ssim"]:.4f}, '
                      f'LPIPS: {epoch_metrics["lpips"]:.4f}, MS-SSIM: {epoch_metrics["msssim"]:.4f}')
                
                # Test random image
                self.test_random_images(epoch)
                
                # Save checkpoint
                is_best = epoch_metrics['psnr'] > self.best_psnr
                if is_best:
                    self.best_psnr = epoch_metrics['psnr']
                
                self.save_checkpoint(epoch, is_best)
            else:
                print(f'Epoch [{epoch+1}/{self.num_epochs}] No valid batches processed.')
        
        # Plot all metrics at the end of training
        plot_metrics(self.metric_history, self.graph_dir)
    
    def denoise_image(self, image_path, output_path, image_size=(256, 256)):
        self.generator.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                denoised_img = self.generator(img_tensor)
            
            denoised_img = denoised_img.squeeze(0).cpu() * 0.5 + 0.5
            denoised_pil = transforms.ToPILImage()(denoised_img)
            denoised_pil.save(output_path)
            print(f"Saved denoised image: {output_path}")
        except Exception as e:
            print(f"Error denoising image {image_path}: {str(e)}")

if __name__ == '__main__':
    try:
        trainer = DenoiseGANTrainer(
            noisy_base_dir='Dataset_Noise',
            clean_dir='Clean_dataset',
            checkpoint_dir='checkpoint',
            graph_dir='graphs',
            test_image_dir='testImage',
            num_epochs=int(input("Enter the number of epochs : ")),
            batch_size=16,
            image_size=(256, 256)
        )
        trainer.train()
        trainer.denoise_image('input_noise.jpg', 'output_denoised.jpg')
    except Exception as e:
        print(f"Error in main execution: {str(e)}")