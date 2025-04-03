import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet18_Weights
from tqdm import tqdm  

# ✅ Force CPU Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Using device: {device}")

# ✅ Dataset Path Check
dataset_path = r"C:\AI_PROJECT\Indoor_waste_dataset\train"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"🚨 Dataset path '{dataset_path}' does not exist. Please check.")

# ✅ Image Preprocessing (Matches ResNet-18 requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# ✅ Load Dataset & Fix Class Issue
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
num_classes = len(dataset.classes)  

# ✅ Split dataset into Train & Validation (80% Train, 20% Validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ✅ DataLoaders (Remove `pin_memory=True` for CPU)
batch_size = 8  # Reduced batch size for CPU efficiency
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ✅ Load Pretrained ResNet18 Model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# ✅ Modify the final layer to match dataset classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  

# ✅ Move Model to CPU
model.to(device)

# ✅ Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)  

# ✅ Training Parameters
epochs = 5
best_val_loss = float("inf")  

# ✅ Train the Model
for epoch in range(epochs):
    model.train()  
    running_loss = 0.0

    # ✅ Training Loop
    progress_bar = tqdm(train_loader, desc=f"🚀 Training Epoch {epoch+1}/{epochs}", leave=False)
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  

        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=running_loss / ((batch_idx + 1) * batch_size))  

    avg_train_loss = running_loss / len(train_loader.dataset)
    print(f"✅ Epoch {epoch+1}/{epochs} | Training Loss: {avg_train_loss:.4f}")

    # ✅ Validation Loop
    model.eval()  
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():  
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)  

            # ✅ Compute Accuracy
            _, predicted = torch.max(outputs, 1)
            total += int(labels.size(0))  
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)  
    accuracy = 100 * correct / total
    print(f"🎯 Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # ✅ Save Best Model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "waste_detection_model_cpu.pth")
        print("🔥 Model saved as 'waste_detection_model_cpu.pth'!")

print("✅ Training Complete! 🚀")
