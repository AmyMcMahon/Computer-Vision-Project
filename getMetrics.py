import torch
import numpy as np
from scipy import linalg
from torchvision import transforms
from PIL import Image
import os

def calculate_fid(real_features, fake_features):
    """Calculate FID score between real and fake image features"""
    # Calculate mean and covariance for real images
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    
    # Calculate mean and covariance for fake images
    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def load_models():
    """Load the trained discriminator and generator models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load discriminator
    discriminator = torch.load('discriminator_final.pth', map_location=device)
    discriminator.eval()
    
    # Load generator
    generator = torch.load('generator_final.pth', map_location=device)
    generator.eval()
    
    return discriminator, generator, device

def extract_features(model, images, device):
    """Extract features from images using the discriminator"""
    features = []
    model.eval()
    
    with torch.no_grad():
        for img in images:
            img = img.unsqueeze(0).to(device)
            # Get features from second-to-last layer
            feat = model.features(img) if hasattr(model, 'features') else model(img)
            features.append(feat.cpu().numpy().flatten())
    
    return np.array(features)

def get_fid_score(real_images_path, num_fake_samples=1000):
    """Calculate FID score between real images and generated images"""
    discriminator, generator, device = load_models()
    
    # Load real images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    real_images = []
    for img_file in os.listdir(real_images_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(real_images_path, img_file))
            img = transform(img)
            real_images.append(img)
    
    # Generate fake images
    fake_images = []
    with torch.no_grad():
        for _ in range(num_fake_samples):
            noise = torch.randn(1, 100, 1, 1).to(device)  # Adjust noise dim as needed
            fake_img = generator(noise)
            fake_images.append(fake_img.squeeze(0).cpu())
    
    # Extract features
    real_features = extract_features(discriminator, real_images, device)
    fake_features = extract_features(discriminator, fake_images, device)
    
    # Calculate FID
    fid_score = calculate_fid(real_features, fake_features)
    
    return fid_score

if __name__ == "__main__":
    # Example usage
    real_images_path = "celeba_gan/img_align_celeba"  # Update this path
    fid = get_fid_score(real_images_path)
    print(f"FID Score: {fid}")