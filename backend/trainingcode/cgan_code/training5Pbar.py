# Set Matplotlib backend to Agg before any imports
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot

import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm for progress bars

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

# Test model function
def test_model(generator, dataset, split, num_samples=3, epoch=None):
    psnr_values, ssim_values = [], []
    for noisy, clean in dataset.take(num_samples):
        noisy = noisy[0] if noisy.shape.rank == 4 else noisy
        clean = clean[0] if clean.shape.rank == 4 else clean
        generated = generator(tf.expand_dims(noisy, 0), training=False)[0]
        clean = tf.cast(clean, tf.float32)
        generated = tf.cast(generated, tf.float32)
        psnr = tf.image.psnr(clean, generated, max_val=2.0)
        ssim = tf.image.ssim(clean, generated, max_val=2.0)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
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
                os.makedirs('TestImg', exist_ok=True)  # Ensure TestImg directory exists
                plt.savefig(f'TestImg/{split}_sampleepoch{epoch}.png', dpi=100)
                plt.close()
            except Exception as e:
                print(f"Error saving plot for {split}_sampleepoch{epoch}.png: {e}")
    return tf.reduce_mean(psnr_values), tf.reduce_mean(ssim_values)

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
    
    val_psnr_history = []
    val_ssim_history = []
    g_loss_history = []
    d_loss_history = []
    
    # Calculate number of batches per epoch
    num_batches = tf.data.experimental.cardinality(train_dataset).numpy()
    if num_batches <= 0:
        num_batches = len(list(train_dataset))  # Fallback if cardinality is unknown
    
    # Progress bar for epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Progress bar for batches
        batch_pbar = tqdm(train_dataset, total=num_batches, desc=f"Epoch {epoch+1} Batches", leave=False)
        for batch in batch_pbar:
            g_loss, d_loss = train_step(batch[0], batch[1], generator, discriminator, gan, g_optimizer, d_optimizer)
            # Update batch progress bar with current losses
            batch_pbar.set_postfix({'G Loss': f'{float(g_loss):.4f}', 'D Loss': f'{float(d_loss):.4f}'})
        
        g_loss_history.append(float(g_loss))
        d_loss_history.append(float(d_loss))
        
        val_psnr, val_ssim = test_model(generator, val_dataset, "val", num_samples=3, epoch=epoch+1)
        val_psnr_history.append(float(val_psnr))
        val_ssim_history.append(float(val_ssim))
        
        print(f'Epoch {epoch+1}, Gen Loss: {g_loss:.4f}, Disc Loss: {d_loss:.4f}, '
              f'Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}')
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            try:
                generator.save(os.path.join(best_chkpoint_dir, f"generator_epoch_{epoch+1}.keras"))
                print(f"Saved checkpoint with PSNR {val_psnr:.2f} at epoch {epoch+1}")
            except Exception as e:
                print(f"Error saving model at epoch {epoch+1}: {e}")
        if ((epoch % 2 == 0) or (epoch == (epochs-1))):
            generator.save(os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.keras"))
            
    return val_psnr_history, val_ssim_history, g_loss_history, d_loss_history

# Main execution
if __name__ == '__main__':
    # Enable mixed precision for GPU optimization
    #from keras import mixed_precision
    #mixed_precision.set_global_policy('mixed_float16')
    
    # Verify GPU availability
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    
    # Initialize models
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    # Load cached dataset
    cache_dir = "./Pre_dataset/cache"
    element_spec = (
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.load(cache_dir, element_spec=element_spec)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_split=0.8, val_split=0.1)
    
    # Apply batching and prefetching
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
    
    # Train the model and collect metrics
    epch = int(input("Enter the number of epoch for the training : ")) 
    try:
        val_psnr_history, val_ssim_history, g_loss_history, d_loss_history = train_gan(
            train_dataset, val_dataset, epochs=epch, generator=generator, 
            discriminator=discriminator, gan=gan, batch_size=batch_size
        )
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    # Plot metrics
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot PSNR
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(val_psnr_history) + 1), val_psnr_history, label='Validation PSNR', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('Validation PSNR over Epochs')
        plt.legend()
        plt.grid(True)
        
        # Plot SSIM
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(val_ssim_history) + 1), val_ssim_history, label='Validation SSIM', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Validation SSIM over Epochs')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs('Graphs', exist_ok=True)  # Ensure Graphs directory exists
        plt.savefig('Graphs/Gmetrics_plot.png', dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error saving metrics plot: {e}")
    
    # Final evaluation on test dataset
    try:
        test_psnr, test_ssim = test_model(generator, test_dataset, "test", num_samples=3, epoch=epch)
        print(f'Test PSNR: {test_psnr:.2f}, Test SSIM: {test_ssim:.4f}')
    except Exception as e:
        print(f"Error during test evaluation: {e}")
