import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations (resize, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit models like ResNet
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Load dataset
dataset_path = r"C:\AI_PROJECT\Indoor_waste_dataset\train"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Check dataset loaded successfully
print(f"Dataset Size: {len(dataset)} images")
print(f"Classes: {dataset.classes}")

# Test DataLoader
images, labels = next(iter(data_loader))
print(f"Batch size: {images.shape}")
