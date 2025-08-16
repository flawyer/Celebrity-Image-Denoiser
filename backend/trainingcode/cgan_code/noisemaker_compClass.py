import os
import os.path 
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

def process_images(input_folder, output_folder):
    """Process all images in the input folder and save noisy versions."""
    # Create output folder if it doesn't exist
    #os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Noise functions and their names
    noise_functions = {
        'gaussian': add_gaussian_noise,
        'salt_pepper': add_salt_pepper_noise,
        'poisson': add_poisson_noise,
        'speckle': add_speckle_noise,
        'uniform': add_uniform_noise
    }

    # Iterate through all files in the input folder
	
		
    for filename in os.listdir(input_folder):
        filename = input_folder+ os.path.sep + filename
        if (os.path.isdir(filename)):
          op_f = os.path.join(output_folder,  filename[2:])
          process_images(filename , op_f)
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in valid_extensions:
            #file_path = os.path.join(input_folder, filename)
            file_path = filename
            print(file_path)
            try:
                # Open the image
                img = Image.open(file_path).convert('RGB')
                
                # Apply each type of noise
                for noise_name, noise_func in noise_functions.items():
                    # Generate noisy image
                    noisy_img = noise_func(img)
                    
                    # Create output filename
                    base_name = os.path.splitext(filename)[0]
                    bName = base_name.split(os.path.sep)[-1]
                    output_filename = f"{bName}{file_ext}"
                    o_folder = output_folder.split(os.path.sep)
                    output_fold = os.path.join(*o_folder[:-1], noise_name,o_folder[-1] )
                    os.makedirs(output_fold, exist_ok=True)
                    #output_path = os.path.join(output_folder, output_filename)
                    output_path = os.path.join(output_fold,output_filename )
                    
                    # Save noisy image
                    noisy_img.save(output_path)
                    print(f"Saved: {output_path}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Specify input and output directories
    input_folder = "./Clean_dataset"  # Change this to your input folder path
    output_folder = "./Dataset_Noise"  # Output folder for noisy images
    
    # Process images
    process_images(input_folder, output_folder)
