# Set Matplotlib backend to Agg before any imports
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot

import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm for progress bars
import lpips  # Import LPIPS for perceptual loss metric
from skimage.metrics import structural_similarity as ssim_multiscale  # For MS-SSIM
import torch  # Import PyTorch for LPIPS compatibility

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex', version='0.1').eval()  # Using AlexNet-based LPIPS, set to evaluation mode
lpips_model = lpips_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(256, 256, 3)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    model.add(layers.Conv2D(3, (3, 3), padding='same', activation='tanh'))
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(256, 256, 3)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Combined GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
perceptual_loss = tf.keras.losses.MeanAbsoluteError()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output, generated_image, target_image):
    gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    perc_loss = perceptual_loss(generated_image, target_image)
    return gan_loss + 100 * perc_loss

# Training step
@tf.function
def train_step(noisy_images, clean_images, generator, discriminator, gan, g_optimizer, d_optimizer):
    discriminator.trainable = True
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        generated_images = generator(noisy_images, training=True)
        
        real_output = discriminator(clean_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        g_loss = generator_loss(fake_output, generated_images, clean_images)
        d_loss = discriminator_loss(real_output, fake_output)
    
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    
    discriminator.trainable = False
    
    return g_loss, d_loss

# Test model function with additional metrics
def test_model(generator, dataset, split, num_samples=3, epoch=None):
    psnr_values, ssim_values, lpips_values, msssim_values = [], [], [], []
    for noisy, clean in dataset.take(num_samples):
        noisy = noisy[0] if noisy.shape.rank == 4 else noisy
        clean = clean[0] if clean.shape.rank == 4 else clean
        generated = generator(tf.expand_dims(noisy, 0), training=False)[0]
        clean = tf.cast(clean, tf.float32)
        generated = tf.cast(generated, tf.float32)
        
        # PSNR
        psnr_val = tf.image.psnr(clean, generated, max_val=2.0)
        psnr_values.append(psnr_val)
        
        # SSIM
        ssim_val = tf.image.ssim(clean, generated, max_val=2.0)
        ssim_values.append(ssim_val)
        
        # LPIPS
        try:
            # Convert TensorFlow tensors to NumPy, then to PyTorch tensors
            clean_np = clean.numpy()
            generated_np = generated.numpy()
            # Rescale to [0, 1] for LPIPS
            clean_np = (clean_np + 1) / 2
            generated_np = (generated_np + 1) / 2
            # Convert to PyTorch tensors: [H, W, C] -> [C, H, W]
            clean_torch = torch.from_numpy(clean_np).permute(2, 0, 1).float()
            generated_torch = torch.from_numpy(generated_np).permute(2, 0, 1).float()
            # Add batch dimension: [C, H, W] -> [1, C, H, W]
            clean_torch = clean_torch.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            generated_torch = generated_torch.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            # Compute LPIPS
            lpips_val = lpips_model(clean_torch, generated_torch)
            lpips_values.append(float(lpips_val))
        except Exception as e:
            print(f"Error computing LPIPS: {e}")
            lpips_values.append(0.0)  # Fallback value
        
        # MS-SSIM
        try:
            msssim_val = ssim_multiscale(clean.numpy(), generated.numpy(), data_range=2.0, channel_axis=2)
            msssim_values.append(msssim_val)
        except Exception as e:
            print(f"Error computing MS-SSIM: {e}")
            msssim_values.append(0.0)  # Fallback value
        
        if epoch is not None:
            try:
                plt.figure(figsize=(5, 2.5))
                plt.subplot(1, 3, 1)
                plt.imshow((noisy + 1) / 2)
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow((generated + 1) / 2)
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.imshow((clean + 1) / 2)
                plt.axis('off')
                os.makedirs('TestImg', exist_ok=True)
                plt.savefig(f'TestImg/{split}_sampleepoch{epoch}.png', dpi=100)
                plt.close()
            except Exception as e:
                print(f"Error saving plot for {split}_sampleepoch{epoch}.png: {e}")
    
    return (tf.reduce_mean(psnr_values), tf.reduce_mean(ssim_values), 
            tf.reduce_mean(lpips_values), tf.reduce_mean(msssim_values))

# Split dataset function
def split_dataset(dataset, train_split=0.8, val_split=0.1):
    data_list = list(dataset)
    if not data_list:
        raise ValueError("Dataset is empty. Cannot split.")
    
    print(f"Total samples: {len(data_list)}")
    
    train_data, temp_data = train_test_split(data_list, train_size=train_split, random_state=None)
    val_data, test_data = train_test_split(temp_data, train_size=val_split/(1-train_split), random_state=None)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    train_dataset = tf.data.Dataset.from_generator(
        lambda: iter(train_data),
        output_types=(tf.float32, tf.float32),
        output_shapes=((256, 256, 3), (256, 256, 3))
    )
    val_dataset = tf.data.Dataset.from_generator(
        lambda: iter(val_data),
        output_types=(tf.float32, tf.float32),
        output_shapes=((256, 256, 3), (256, 256, 3))
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: iter(test_data),
        output_types=(tf.float32, tf.float32),
        output_shapes=((256, 256, 3), (256, 256, 3))
    )
    
    return train_dataset, val_dataset, test_dataset

# Training loop
def train_gan(train_dataset, val_dataset, epochs, generator, discriminator, gan, batch_size=32, checkpoint_dir="./checkpoints"):
    g_optimizer = tf.keras.optimizers.Adam(1e-4)
    d_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_chkpoint_dir = os.path.join(checkpoint_dir, "best")
    os.makedirs(best_chkpoint_dir, exist_ok=True)
    best_val_psnr = 0.0
    
    val_psnr_history, val_ssim_history, val_lpips_history, val_msssim_history = [], [], [], []
    g_loss_history, d_loss_history = [], []
    
    num_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    if num_batches <= 0:
        num_batches = len(list(train_dataset))
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        batch_pbar = tqdm(train_dataset, total=num_batches, desc=f"Epoch {epoch+1} Batches", leave=False)
        for batch in batch_pbar:
            g_loss, d_loss = train_step(batch[0], batch[1], generator, discriminator, gan, g_optimizer, d_optimizer)
            batch_pbar.set_postfix({'G Loss': f'{float(g_loss):.4f}', 'D Loss': f'{float(d_loss):.4f}'})
        
        g_loss_history.append(float(g_loss))
        d_loss_history.append(float(d_loss))
        
        val_psnr, val_ssim, val_lpips, val_msssim = test_model(generator, val_dataset, "val", num_samples=3, epoch=epoch+1)
        val_psnr_history.append(float(val_psnr))
        val_ssim_history.append(float(val_ssim))
        val_lpips_history.append(float(val_lpips))
        val_msssim_history.append(float(val_msssim))
        
        print(f'Epoch {epoch+1}, Gen Loss: {g_loss:.4f}, Disc Loss: {d_loss:.4f}, '
              f'Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}, '
              f'Val LPIPS: {val_lpips:.4f}, Val MS-SSIM: {val_msssim:.4f}')
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            try:
                generator.save(os.path.join(best_chkpoint_dir, f"generator_epoch_{epoch+1}.keras"))
                print(f"Saved checkpoint with PSNR {val_psnr:.2f} at epoch {epoch+1}")
            except Exception as e:
                print(f"Error saving model at epoch {epoch+1}: {e}")
        if ((epoch % 2 == 0) or (epoch == (epochs-1))):
            generator.save(os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.keras"))
            
    return val_psnr_history, val_ssim_history, val_lpips_history, val_msssim_history, g_loss_history, d_loss_history

# Main execution
if __name__ == '__main__':
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    cache_dir = "./Pre_dataset/cache"
    element_spec = (
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.load(cache_dir, element_spec=element_spec)
    
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_split=0.8, val_split=0.1)
    
    batch_size = 8
    train_dataset = train_dataset.shuffle(buffer_size=500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for noisy, clean in train_dataset.take(1):
        print("Train noisy shape:", noisy.shape)
        print("Train clean shape:", clean.shape)
    for noisy, clean in val_dataset.take(1):
        print("Val noisy shape:", noisy.shape)
        print("Val clean shape:", clean.shape)
    
    epch = int(input("Enter the number of epoch for the training : "))
    try:
        val_psnr_history, val_ssim_history, val_lpips_history, val_msssim_history, g_loss_history, d_loss_history = train_gan(
            train_dataset, val_dataset, epochs=epch, generator=generator, 
            discriminator=discriminator, gan=gan, batch_size=batch_size
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    # Plot individual metrics
    try:
        os.makedirs('Graphs', exist_ok=True)
        
        # PSNR Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_psnr_history) + 1), val_psnr_history, label='Validation PSNR', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('Validation PSNR over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('Graphs/psnr.png', dpi=100)
        plt.close()
        
        # SSIM Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_ssim_history) + 1), val_ssim_history, label='Validation SSIM', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Validation SSIM over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('Graphs/ssim.png', dpi=100)
        plt.close()
        
        # LPIPS Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_lpips_history) + 1), val_lpips_history, label='Validation LPIPS', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('LPIPS')
        plt.title('Validation LPIPS over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('Graphs/lpips.png', dpi=100)
        plt.close()
        
        # MS-SSIM Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_msssim_history) + 1), val_msssim_history, label='Validation MS-SSIM', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('MS-SSIM')
        plt.title('Validation MS-SSIM over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('Graphs/msssim.png', dpi=100)
        plt.close()
        
        # Generator Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(g_loss_history) + 1), g_loss_history, label='Generator Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('Graphs/g_loss.png', dpi=100)
        plt.close()
        
        # Discriminator Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(d_loss_history) + 1), d_loss_history, label='Discriminator Loss', color='cyan')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('Graphs/d_loss.png', dpi=100)
        plt.close()
        
        # Summary Plot
        plt.figure(figsize=(15, 10))
        
        # PSNR
        plt.subplot(2, 3, 1)
        plt.plot(range(1, len(val_psnr_history) + 1), val_psnr_history, label='PSNR', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('Validation PSNR')
        plt.legend()
        plt.grid(True)
        
        # SSIM
        plt.subplot(2, 3, 2)
        plt.plot(range(1, len(val_ssim_history) + 1), val_ssim_history, label='SSIM', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Validation SSIM')
        plt.legend()
        plt.grid(True)
        
        # LPIPS
        plt.subplot(2, 3, 3)
        plt.plot(range(1, len(val_lpips_history) + 1), val_lpips_history, label='LPIPS', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('LPIPS')
        plt.title('Validation LPIPS')
        plt.legend()
        plt.grid(True)
        
        # MS-SSIM
        plt.subplot(2, 3, 4)
        plt.plot(range(1, len(val_msssim_history) + 1), val_msssim_history, label='MS-SSIM', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('MS-SSIM')
        plt.title('Validation MS-SSIM')
        plt.legend()
        plt.grid(True)
        
        # Generator Loss
        plt.subplot(2, 3, 5)
        plt.plot(range(1, len(g_loss_history) + 1), g_loss_history, label='Generator Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Loss')
        plt.legend()
        plt.grid(True)
        
        # Discriminator Loss
        plt.subplot(2, 3, 6)
        plt.plot(range(1, len(d_loss_history) + 1), d_loss_history, label='Discriminator Loss', color='cyan')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('Graphs/summary.png', dpi=100)
        plt.close()
        
    except Exception as e:
        print(f"Error saving metrics plots: {e}")
    
    # Final evaluation on test dataset
    try:
        test_psnr, test_ssim, test_lpips, test_msssim = test_model(generator, test_dataset, "test", num_samples=3, epoch=epch)
        print(f'Test PSNR: {test_psnr:.2f}, Test SSIM: {test_ssim:.4f}, '
              f'Test LPIPS: {test_lpips:.4f}, Test MS-SSIM: {test_msssim:.4f}')
    except Exception as e:
        print(f"Error during test evaluation: {e}")