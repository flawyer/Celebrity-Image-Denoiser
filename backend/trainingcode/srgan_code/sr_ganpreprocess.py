import tensorflow as tf
import os
import glob
import shutil
from PIL import Image, ImageFile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_and_preprocess(noisy_path, clean_path, target_size=(256, 256)):
    """Load and preprocess images to 256x256, normalize to [-1, 1]."""
    try:
        noisy_path_str = noisy_path.numpy().decode('utf-8') if isinstance(noisy_path, tf.Tensor) else noisy_path
        clean_path_str = clean_path.numpy().decode('utf-8') if isinstance(clean_path, tf.Tensor) else clean_path

        noisy_image = tf.io.read_file(noisy_path)
        clean_image = tf.io.read_file(clean_path)
        noisy_image = tf.image.decode_image(noisy_image, channels=3, dtype=tf.float32, expand_animations=False)
        clean_image = tf.image.decode_image(clean_image, channels=3, dtype=tf.float32, expand_animations=False)

        noisy_shape = tf.shape(noisy_image)
        clean_shape = tf.shape(clean_image)
        if tf.reduce_any(noisy_shape < 1) or tf.reduce_any(clean_shape < 1):
            raise ValueError(f"Invalid shape for {noisy_path_str} or {clean_path_str}: {noisy_shape}, {clean_shape}")

        noisy_image = tf.image.resize(noisy_image, target_size, method='lanczos3')
        clean_image = tf.image.resize(clean_image, target_size, method='lanczos3')

        noisy_image = (noisy_image - 127.5) / 127.5
        clean_image = (clean_image - 127.5) / 127.5
        return noisy_image, clean_image
    except Exception as e:
        print(f"Negating invalid image pair {noisy_path_str}, {clean_path_str}: {str(e)}")
        try:
            if os.path.exists(noisy_path_str):
                os.remove(noisy_path_str)
                print(f"Deleted: {noisy_path_str}")
            if os.path.exists(clean_path_str):
                os.remove(clean_path_str)
                print(f"Deleted: {clean_path_str}")
        except Exception as rm_e:
            print(f"Error deleting files: {str(rm_e)}")
        return None

def prepare_data(noisy_dirs, clean_dir, cache_dir="Pre_dataset/cache"):
    """Create dataset with image pairs, negating incomplete or invalid pairs."""
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    clean_images = {}
    deleted_clean = []
    for root, _, files in os.walk(clean_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png')):
                clean_path = os.path.join(root, filename)
                try:
                    with Image.open(clean_path) as img:
                        img.verify()
                    celeb_folder = os.path.relpath(root, clean_dir)
                    clean_name = os.path.splitext(filename)[0]
                    clean_images.setdefault(celeb_folder, {})[clean_name] = clean_path
                except Exception as e:
                    print(f"Deleting corrupted clean image {clean_path}: {str(e)}")
                    try:
                        os.remove(clean_path)
                        deleted_clean.append(clean_path)
                        print(f"Deleted: {clean_path}")
                    except Exception as rm_e:
                        print(f"Error deleting {clean_path}: {str(rm_e)}")
                    continue

    noisy_images = {}
    deleted_noisy = []
    for noise_dir in noisy_dirs:
        for root, _, files in os.walk(noise_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.png')):
                    noisy_path = os.path.join(root, filename)
                    try:
                        with Image.open(noisy_path) as img:
                            img.verify()
                        celeb_folder = os.path.normpath(os.path.relpath(root, noise_dir))
                        noisy_name = os.path.splitext(filename)[0]
                        noisy_images.setdefault(noise_dir, {}).setdefault(celeb_folder, {})[noisy_name] = noisy_path
                    except Exception as e:
                        print(f"Deleting corrupted noisy image {noisy_path}: {str(e)}")
                        try:
                            os.remove(noisy_path)
                            deleted_noisy.append((noise_dir, noisy_path))
                            print(f"Deleted: {noisy_path}")
                        except Exception as rm_e:
                            print(f"Error deleting {noisy_path}: {str(rm_e)}")
                            continue

    image_pairs = []
    negated_pairs = []
    for celeb_folder in sorted(clean_images.keys()):
        for clean_name, clean_path in clean_images[celeb_folder].items():
            # Check if all noisy versions exist
            valid = True
            noisy_paths = []
            for noise_dir in noisy_dirs:
                if (noise_dir not in noisy_images or
                    celeb_folder not in noisy_images[noise_dir] or
                    clean_name not in noisy_images[noise_dir][celeb_folder]):
                    valid = False
                    negated_pairs.append((clean_path, noise_dir, celeb_folder, clean_name))
                    break
                noisy_paths.append(noisy_images[noise_dir][celeb_folder][clean_name])
            # If all noisy versions exist, add pairs
            if valid:
                for noisy_path in noisy_paths:
                    image_pairs.append((noisy_path, clean_path))
            else:
                # Negate pair: delete clean image and all existing noisy versions
                print(f"Negating image {clean_path}: Missing noisy version")
                try:
                    if os.path.exists(clean_path):
                        os.remove(clean_path)
                        deleted_clean.append(clean_path)
                        print(f"Deleted: {clean_path}")
                    for noise_dir in noisy_dirs:
                        if (noise_dir in noisy_images and
                            celeb_folder in noisy_images[noise_dir] and
                            clean_name in noisy_images[noise_dir][celeb_folder]):
                            noisy_path = noisy_images[noise_dir][celeb_folder][clean_name]
                            if os.path.exists(noisy_path):
                                os.remove(noisy_path)
                                deleted_noisy.append((noise_dir, noisy_path))
                                print(f"Deleted: {noisy_path}")
                except Exception as rm_e:
                    print(f"Error deleting images for {clean_path}: {str(rm_e)}")

    if deleted_clean:
        print(f"Deleted {len(deleted_clean)} clean images:")
        for img in deleted_clean[:5]:
            print(f"Deleted: {img}")
        if len(deleted_clean) > 5:
            print(f"... and {len(deleted_clean) - 5} more")

    if deleted_noisy:
        print(f"Deleted {len(deleted_noisy)} noisy images:")
        for noise_dir, img in deleted_noisy[:5]:
            print(f"Deleted: {img} ({noise_dir})")
        if len(deleted_noisy) > 5:
            print(f"... and {len(deleted_noisy) - 5} more")

    if negated_pairs:
        print(f"Negated {len(negated_pairs)} pairs due to missing noisy versions:")
        for clean_path, noise_dir, celeb_folder, clean_name in negated_pairs[:5]:
            print(f"Negated: {clean_path} (missing {noise_dir}/{celeb_folder}/{clean_name}.jpg)")
        if len(negated_pairs) > 5:
            print(f"... and {len(negated_pairs) - 5} more")

    if not image_pairs:
        raise ValueError("No valid image pairs found. Check if images are valid and paths are correct.")

    print(f"Total valid pairs: {len(image_pairs)}")

    dataset = tf.data.Dataset.from_tensor_slices(([p[0] for p in image_pairs], [p[1] for p in image_pairs]))

    @tf.py_function(Tout=[tf.float32, tf.float32])
    def wrapped_load_and_preprocess(noisy_path, clean_path):
        result = load_and_preprocess(noisy_path, clean_path, target_size=(256, 256))
        if result is None:
            return [tf.zeros((256, 256, 3), dtype=tf.float32), tf.zeros((256, 256, 3), dtype=tf.float32)]
        return result

    dataset = dataset.map(
        wrapped_load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.filter(lambda x, y: tf.reduce_all(tf.shape(x) == [256, 256, 3]) and tf.reduce_all(tf.shape(y) == [256, 256, 3]))
    dataset = dataset.take(len(image_pairs)).cache(cache_dir).repeat()
    dataset = dataset.shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)

    return dataset

if __name__ == '__main__':
    noisy_dirs = [
        "./Dataset_Noise/gaussian",
        "./Dataset_Noise/salt_pepper",
        "./Dataset_Noise/uniform",
        "./Dataset_Noise/poisson",
        "./Dataset_Noise/speckle"
    ]
    clean_dir = "./Clean_dataset"
    dataset = prepare_data(noisy_dirs, clean_dir)
    count = 0
    for noisy, clean in dataset:
        if count == 0:
            print("Noisy shape:", noisy.shape)
            print("Clean shape:", clean.shape)
        count += 1
        if count >= len(noisy_dirs) * 1000:  # Limit for testing
            break
    print(f"Processed {count} pairs for testing")
