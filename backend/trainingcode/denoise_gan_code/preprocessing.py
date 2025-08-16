import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split

class DenoiseDataset(Dataset):
    def __init__(self, noisy_base_dir, clean_dir, noise_types, image_size=(256, 256), test_split=0.2):
        self.noisy_base_dir = noisy_base_dir
        self.clean_dir = clean_dir
        self.noise_types = noise_types
        self.image_size = image_size
        self.image_pairs = []
        self.test_image_pairs = []
        
        # Collect all image pairs and split into train/test
        all_pairs = []
        for noise_type in noise_types:
            noise_dir = os.path.join(noisy_base_dir, noise_type)
            if os.path.exists(noise_dir):
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

if __name__ == '__main__':
    dataset = DenoiseDataset(
        noisy_base_dir='Dataset_Noise',
        clean_dir='Clean_dataset',
        noise_types=['gaussian', 'salt_pepper', 'speckle', 'poisson', 'uniform'],
        image_size=(256, 256),
        test_split=0.2
    )
    print(f"Dataset ready with {len(dataset)} training pairs.")