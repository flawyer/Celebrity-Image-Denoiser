import os
import numpy as np
from PIL import Image
import skimage.util
from pathlib import Path

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image."""
    img_array = np.array(image).astype(float)
    noise = np.random.normal(mean, sigma, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def add_salt_pepper_noise(image, amount=0.05):
    """Add salt and pepper noise to an image."""
    img_array = np.array(image)
    noisy_img = skimage.util.random_noise(img_array, mode='s&p', amount=amount)
    noisy_img = (noisy_img * 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def add_poisson_noise(image):
    """Add Poisson noise to an image."""
    img_array = np.array(image)
    noisy_img = skimage.util.random_noise(img_array, mode='poisson')
    noisy_img = (noisy_img * 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def add_speckle_noise(image, sigma=0.1):
    """Add speckle noise to an image."""
    img_array = np.array(image).astype(float)
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy_img = img_array + img_array * noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def add_uniform_noise(image, low=-50, high=50):
    """Add uniform noise to an image."""
    img_array = np.array(image).astype(float)
    noise = np.random.uniform(low, high, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def process_images(input_folder, output_noisy_folder, output_clean_folder, lr_size=(64, 64), hr_size=(256, 256)):
    """Process images to generate noisy LR and clean HR images for SRGAN."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    noise_functions = {
        'gaussian': add_gaussian_noise,
        'salt_pepper': add_salt_pepper_noise,
        'poisson': add_poisson_noise,
        'speckle': add_speckle_noise,
        'uniform': add_uniform_noise
    }

    def process_directory(current_input_path, relative_path):
        """Recursively process directories and files."""
        # Create corresponding output directories
        noisy_base_dir = os.path.join(output_noisy_folder, relative_path)
        clean_person_dir = os.path.join(output_clean_folder, relative_path)
        os.makedirs(clean_person_dir, exist_ok=True)

        print(f"Processing directory: {current_input_path}")
        print(f"Clean output directory: {clean_person_dir}")

        # Process each item in the current directory
        for item in os.listdir(current_input_path):
            item_path = os.path.join(current_input_path, item)
            if os.path.isdir(item_path):
                # Recursively process subdirectories
                new_relative_path = os.path.join(relative_path, item)
                process_directory(item_path, new_relative_path)
            elif os.path.isfile(item_path):
                file_ext = os.path.splitext(item)[1].lower()
                if file_ext in valid_extensions:
                    print(f"Processing file: {item_path}")
                    try:
                        # Open and process the image
                        img = Image.open(item_path).convert('RGB')

                        # Resize for HR clean image
                        hr_img = img.resize(hr_size, resample=Image.Resampling.BICUBIC)
                        clean_output_path = os.path.join(clean_person_dir, item)
                        hr_img.save(clean_output_path)
                        print(f"Saved clean HR image: {clean_output_path}")

                        # Resize for LR image before adding noise
                        lr_img = img.resize(lr_size, resample=Image.Resampling.BICUBIC)

                        # Apply each type of noise to LR image
                        for noise_name, noise_func in noise_functions.items():
                            noisy_img = noise_func(lr_img)
                            noise_dir = os.path.join(output_noisy_folder, noise_name, relative_path)
                            os.makedirs(noise_dir, exist_ok=True)
                            noisy_output_path = os.path.join(noise_dir, item)
                            noisy_img.save(noisy_output_path)
                            print(f"Saved noisy LR image: {noisy_output_path}")

                    except Exception as e:
                        print(f"Error processing {item_path}: {e}")

    # Start processing from the input folder
    input_path = Path(input_folder)
    process_directory(input_path, "")

if __name__ == "__main__":
    # Specify input and output directories
    input_folder = "./Clean_dataset"  # Input folder with clean images
    output_noisy_folder = "./Dataset_Noise"  # Output folder for noisy LR images
    output_clean_folder = "./Clean_dataset"  # Output folder for clean HR images

    # Process images with LR size 64x64 and HR size 256x256
    process_images(
        input_folder=input_folder,
        output_noisy_folder=output_noisy_folder,
        output_clean_folder=output_clean_folder,
        lr_size=(64, 64),  # Low-resolution size for SRGAN
        hr_size=(256, 256)  # High-resolution size for SRGAN
    )