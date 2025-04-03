import torch
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# ✅ Initialize Model (Same Architecture as Training)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 5)  # Adjust for 5 classes

# ✅ Load Trained Weights
model.load_state_dict(torch.load("waste_detection_model_cpu.pth", map_location=torch.device("cpu")))
model.eval()  # Put model in evaluation mode

# ✅ Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Class Names (Modify if Needed)
class_names = ['cardboard', 'metal', 'paper', 'plastic', 'trash']
confidence_threshold = 0.60  # Adjust as needed (e.g., 60% confidence)

# ✅ Function to Select and Predict Image
def select_and_predict():
    # 🖼 Open File Dialog to Select Image
    root = tk.Tk()
    root.withdraw()  # Hide main window
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])

    if not file_path:
        print("No file selected.")
        return

    # ✅ Load and Process Image
    image = Image.open(file_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # ✅ Predict Class
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence_score = probabilities[predicted_class].item()

    # ✅ Check Confidence and Print Result
    if confidence_score < confidence_threshold:
        print("Predicted: Unknown Object")
    else:
        print(f"Predicted: {class_names[predicted_class]} ({confidence_score*100:.2f}%)")

# ✅ Run Prediction
select_and_predict()
