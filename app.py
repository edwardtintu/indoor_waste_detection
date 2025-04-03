import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ✅ Define class labels
class_labels = ["cardboard", "metal", "paper", "plastic", "trash"]

# ✅ Load the model
st.write("🔄 Loading model...")

try:
    model = models.resnet18()  # Change this if a different architecture was used
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_labels))  # Set number of classes

    # ✅ Load only the state_dict (weights)
    model.load_state_dict(torch.load("waste_detection_model_cpu.pth", map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# 🔄 Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🚀 Streamlit UI
st.title("♻️ Waste Classification App")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # ✅ Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Preprocess Image
    image = transform(image).unsqueeze(0)

    # ✅ Predict
    with torch.no_grad():
        output = model(image)
        confidence = torch.nn.functional.softmax(output, dim=1)[0] * 100
        predicted_class_idx = output.argmax().item()

    # ✅ Check if prediction is valid
    if 0 <= predicted_class_idx < len(class_labels):
        predicted_label = class_labels[predicted_class_idx]
        confidence_score = confidence[predicted_class_idx].item()
    else:
        predicted_label = "Unknown Object"
        confidence_score = 0.00

    # ✅ Display Result
    st.success(f"Predicted: **{predicted_label}** ({confidence_score:.2f}%)")
