import tensorflow as tf
import cv2
import numpy as np
import os
import glob
import shutil

# Optional: Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loads images, resizes to (256,256), and normalizes to [-1,1]
def load_and_preprocess(noisy_path, clean_path):
    noisy_image = cv2.imread(noisy_path)
    clean_image = cv2.imread(clean_path)
    if noisy_image is None or clean_image is None:
        raise ValueError(f"Failed to load images: {noisy_path}, {clean_path}")
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
    clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
    noisy_image = cv2.resize(noisy_image, (256, 256)).astype(np.float32)
    clean_image = cv2.resize(clean_image, (256, 256)).astype(np.float32)
    noisy_image = (noisy_image - 127.5) / 127.5
    clean_image = (clean_image - 127.5) / 127.5
    return noisy_image, clean_image

# Create matching pairs of images and save as a dataset
def prepare_data(noisy_dirs, clean_dir, cache_dir="Pre_dataset/cache"):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    clean_images = {}
    for celeb_folder in os.listdir(clean_dir):
        celeb_path = os.path.join(clean_dir, celeb_folder)
        if not os.path.isdir(celeb_path):
            continue
        clean_paths = glob.glob(os.path.join(celeb_path, "*.*"))
        for clean_path in clean_paths:
            if not clean_path.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            clean_name = os.path.splitext(os.path.basename(clean_path))[0]
            clean_images.setdefault(celeb_folder, {})[clean_name] = clean_path

    noisy_images = {}
    for noise_dir in noisy_dirs:
        for celeb_folder in os.listdir(noise_dir):
            celeb_path = os.path.join(noise_dir, celeb_folder)
            if not os.path.isdir(celeb_path):
                continue
            noisy_paths = glob.glob(os.path.join(celeb_path, "*.*"))
            for noisy_path in noisy_paths:
                if not noisy_path.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                noisy_name = os.path.splitext(os.path.basename(noisy_path))[0]
                noisy_images.setdefault(noise_dir, {}).setdefault(celeb_folder, {})[noisy_name] = noisy_path

    image_pairs = []
    unmatched_clean = []
    unmatched_noisy = []

    for celeb_folder in clean_images:
        for clean_name, clean_path in clean_images[celeb_folder].items():
            found = False
            for noise_dir in noisy_dirs:
                if (noise_dir in noisy_images and
                    celeb_folder in noisy_images[noise_dir] and
                    clean_name in noisy_images[noise_dir][celeb_folder]):
                    noisy_path = noisy_images[noise_dir][celeb_folder][clean_name]
                    image_pairs.append((noisy_path, clean_path))
                    found = True
            if not found:
                unmatched_clean.append((celeb_folder, clean_name, clean_path))

    for noise_dir in noisy_images:
        for celeb_folder in noisy_images[noise_dir]:
            for noisy_name, noisy_path in noisy_images[noise_dir][celeb_folder].items():
                if (celeb_folder not in clean_images or
                    noisy_name not in clean_images[celeb_folder]):
                    unmatched_noisy.append((noise_dir, celeb_folder, noisy_name, noisy_path))

    if not image_pairs:
        raise ValueError("No valid image pairs found.")

    print(f"Total pairs: {len(image_pairs)}")
    if unmatched_clean:
        print(f"Unmatched clean images: {len(unmatched_clean)}")
        print("Sample unmatched clean:", unmatched_clean[:5])
    if unmatched_noisy:
        print(f"Unmatched noisy images: {len(unmatched_noisy)}")
        print("Sample unmatched noisy:", unmatched_noisy[:5])
    print("Sample pairs:", image_pairs[:5])

    def image_generator():
        for noisy_path, clean_path in image_pairs:
            yield load_and_preprocess(noisy_path, clean_path)

    dataset = tf.data.Dataset.from_generator(
        image_generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        )
    )

    dataset = dataset.shuffle(buffer_size=5000)

    print(f"Saving dataset to {cache_dir}")
    tf.data.Dataset.save(dataset, cache_dir)

    return dataset

if __name__ == '__main__':
    noisy_dirs = [
        "Dataset_Noise/gaussian",
        "Dataset_Noise/salt_pepper",
        "Dataset_Noise/uniform",
        "Dataset_Noise/poisson",
        "Dataset_Noise/speckle"
    ]
    clean_dir = "./Clean_dataset"
    dataset = prepare_data(noisy_dirs, clean_dir)

    for noisy, clean in dataset.take(1):
        print("Noisy shape:", noisy.shape)
        print("Clean shape:", clean.shape)
