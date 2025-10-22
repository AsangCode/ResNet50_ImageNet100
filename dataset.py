import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Get dataset directory from environment variable or use default
DATA_DIR = os.getenv('DATASET_PATH', '/data/dataset/imagenet100')  # Use environment variable or default to /data path

class ImageNet100(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Set up paths
        if self.train:
            self.image_dir = os.path.join(root, 'train')
        else:
            self.image_dir = os.path.join(root, 'val')
        
        self.images = []
        self.labels = []
        
        # Load class mapping
        self.class_to_idx = {}
        class_dirs = sorted(os.listdir(self.image_dir))  # Sort to ensure consistent class indices
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir] = idx
            
            # Load images for this class
            class_path = os.path.join(self.image_dir, class_dir)
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_path, img_file))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and convert image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def get_data_loaders(batch_size=128, num_workers=2):
    """Create train and validation data loaders"""
    # Enable memory pinning
    torch.backends.cudnn.benchmark = True
    
    # Define transforms for ImageNet (224x224)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the shorter side to 256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageNet100(DATA_DIR, train=True, transform=train_transform)
    val_dataset = ImageNet100(DATA_DIR, train=False, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
