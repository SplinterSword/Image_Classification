import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import sys
from PIL import Image

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_cifar10_model():
    model_path = os.path.join("src/models", "cifar10_cnn_model.keras")
    try:
        return load_model(model_path)
    except Exception as e:
        st.warning(f"Error loading model: {str(e)}")
        return None

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def preprocess_image(img, target_size=(32, 32)):
    """Preprocess the image to match model's expected formatting."""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def predict_image(model, img_array):
    """Make prediction and return class and confidence."""
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    return predicted_class, confidence, predictions[0]

def main():
    st.title("🖼️ CIFAR-10 Image Classifier")
    st.markdown("---")
    
    # Warning about CIFAR-10 classes
    with st.expander("⚠️ Important Note", expanded=True):
        st.warning("""
        This classifier is trained on the CIFAR-10 dataset and works best with images of the following categories:
        - ✈️ Airplane
        - 🚗 Automobile
        - 🐦 Bird
        - 🐱 Cat
        - 🦌 Deer
        - 🐶 Dog
        - 🐸 Frog
        - 🐴 Horse
        - 🚢 Ship
        - 🚚 Truck
        
        For best results, use clear images that closely match these categories.
        """)
    
    # Load model
    model = load_cifar10_model()
    if model is None:
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess and predict
        with st.spinner("Analyzing image..."):
            try:
                # Preprocess the image
                img_array = preprocess_image(image_display)
                
                # Make prediction
                predicted_class_idx, confidence, all_predictions = predict_image(model, img_array)
                predicted_class = class_names[predicted_class_idx]
                
                # Display results
                st.success(f"Prediction: **{predicted_class.upper()}**")
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Show all class probabilities
                st.subheader("Class Probabilities")
                prob_data = {"Class": class_names, "Probability": all_predictions}
                st.bar_chart(
                    data=prob_data,
                    x="Class",
                    y="Probability",
                    use_container_width=True,
                    height=400
                )
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    else:
        # Show example images when no file is uploaded
        st.info("👆 Please upload an image to get started!")
        
        # Display example images
        st.subheader("Example Images")
        col1, col2, col3 = st.columns(3)
        
        example_images = [
            ("src/plane.jpg", "Airplane"),
            ("src/car.jpg", "Automobile"),
            ("src/deer.jpg", "Deer")
        ]
        
        for img_path, caption in example_images:
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    with col1 if img_path == "src/plane.jpg" else (col2 if img_path == "src/car.jpg" else col3):
                        st.image(img, caption=caption, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load example image {img_path}")

if __name__ == "__main__":
    main()
