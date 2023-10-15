import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import requests

# Title of the app
st.title("Car Technical Verification by Alhorethm")

# Load a pre-trained PyTorch model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load('with_shapes.pth', map_location=torch.device('cpu') )
    model.eval()
    return model

model = load_model()

# Helper function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Input: File Uploader for the image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Output: Display the uploaded image and model prediction
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        st.write("Classifying...")

        # Preprocess the image
        input_image = preprocess_image(image)

        # Make a prediction
        with torch.no_grad():
            output = model(input_image)
            _, predicted_class = output.max(1)

        # Display the prediction
        st.write(f"Predicted class: {predicted_class.item()}")
