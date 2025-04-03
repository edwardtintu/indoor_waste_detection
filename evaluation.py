import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ✅ Load the Trained Model
model = models.resnet18()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 5)  # 5 classes

model.load_state_dict(torch.load("waste_detection_model.pth"))
model.eval()

# ✅ Move Model to GPU if Available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Define Test Dataset
test_path = r"C:\AI_PROJECT\Indoor_waste_dataset\test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ✅ Evaluate the Model
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ✅ Calculate Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Print Classification Report
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
