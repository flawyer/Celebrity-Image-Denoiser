import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the Denoise Generator
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
        e1 = self.down1(x)
        p1 = self.pool1(e1)
        e2 = self.down2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.upconv2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.upconv1(d1)
        return torch.tanh(d1)

def enhance_images(checkpoint_path, input_dir='testNoise', output_dir='testOp', image_size=(256, 256)):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize and load model
    generator = DenoiseGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # Changed to weights_only=False
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Process each image in input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # Load and preprocess image
                img = Image.open(input_path).convert('RGB').resize(image_size, resample=Image.Resampling.BICUBIC)
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Generate denoised image
                with torch.no_grad():
                    denoised_tensor = generator(img_tensor)
                
                # Post-process and save
                denoised_img = denoised_tensor.squeeze(0).cpu() * 0.5 + 0.5
                denoised_pil = transforms.ToPILImage()(denoised_img)
                denoised_pil.save(output_path)
                print(f"Saved denoised image: {output_path}")
                
            except Exception as e:
                print(f"Error processing image {input_path}: {str(e)}")

if __name__ == '__main__':
    # Specify the path to the pre-trained checkpoint
    checkpoint_path = 'checkpoints/denoise_epoch_499.pth'  # Adjust based on your training epoch
    enhance_images(checkpoint_path, input_dir='testNoise', output_dir='testOp', image_size=(256, 256))
