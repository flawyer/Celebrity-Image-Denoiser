import os
import numpy as np
from PIL import Image

# Noise Generation Functions
def add_gaussian_noise(image_array, mean=0, sigma=25):
    """Add Gaussian noise to the image."""
    noise = np.random.normal(mean, sigma, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_pepper_noise(image_array, salt_prob=0.02, pepper_prob=0.02):
    """Add Salt & Pepper noise to the image."""
    noisy_image = image_array.copy()
    total_pixels = image_array.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image_array.shape]
    noisy_image[coords[0], coords[1], :] = 255
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image_array.shape]
    noisy_image[coords[0], coords[1], :] = 0
    return noisy_image

def add_speckle_noise(image_array, mean=0, sigma=0.1):
    """Add Speckle noise to the image."""
    noise = image_array * np.random.normal(mean, sigma, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_poisson_noise(image_array):
    """Add Poisson noise to the image."""
    noisy_image = np.random.poisson(image_array).astype(np.uint8)
    return noisy_image

def add_uniform_noise(image_array, low=0, high=25):
    """Add Uniform noise to the image."""
    noise = np.random.uniform(low, high, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return noisy_image

class NoiseGenerator:
    def __init__(self, clean_dir, noisy_base_dir, noise_types=['gaussian', 'salt_pepper', 'speckle', 'poisson', 'uniform'], image_size=(256, 256)):
        self.clean_dir = clean_dir
        self.noisy_base_dir = noisy_base_dir
        self.noise_types = noise_types
        self.image_size = image_size
        
        # Create noisy directories if they don't exist
        for noise_type in noise_types:
            os.makedirs(os.path.join(noisy_base_dir, noise_type), exist_ok=True)

    def generate_noisy_dataset(self):
        """Generate noisy versions of clean images and save them."""
        for person_dir in os.listdir(self.clean_dir):
            person_clean_dir = os.path.join(self.clean_dir, person_dir)
            if os.path.isdir(person_clean_dir):
                for filename in os.listdir(person_clean_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        clean_path = os.path.join(person_clean_dir, filename)
                        clean_img = Image.open(clean_path).convert('RGB')
                        clean_array = np.array(clean_img.resize(self.image_size, resample=Image.Resampling.BICUBIC))
                        
                        for noise_type in self.noise_types:
                            noisy_dir = os.path.join(self.noisy_base_dir, noise_type, person_dir)
                            os.makedirs(noisy_dir, exist_ok=True)
                            noisy_path = os.path.join(noisy_dir, filename)
                            
                            if noise_type == 'gaussian':
                                noisy_array = add_gaussian_noise(clean_array)
                            elif noise_type == 'salt_pepper':
                                noisy_array = add_salt_pepper_noise(clean_array)
                            elif noise_type == 'speckle':
                                noisy_array = add_speckle_noise(clean_array)
                            elif noise_type == 'poisson':
                                noisy_array = add_poisson_noise(clean_array)
                            elif noise_type == 'uniform':
                                noisy_array = add_uniform_noise(clean_array)
                            
                            noisy_img = Image.fromarray(noisy_array)
                            noisy_img.save(noisy_path)
                            print(f"Saved noisy image: {noisy_path}")

if __name__ == '__main__':
    generator = NoiseGenerator(
        clean_dir='Clean_dataset',
        noisy_base_dir='Dataset_Noise',
        noise_types=['gaussian', 'salt_pepper', 'speckle', 'poisson', 'uniform'],
        image_size=(256, 256)
    )
    generator.generate_noisy_dataset()