import streamlit as st
import torch
import torchvision
import torchmetrics
import pytorch_lightning as pl
import numpy as np
import cv2
import time
import pydicom
import nibabel as nib
import io
from torchvision import transforms
from PIL import Image

# Load the trained model
class PneumoniaModel(pl.LightningModule):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.val_outputs = []

    def forward(self, data):
        return self.model(data)

    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:, 0]
        loss = self.loss_fn(pred, label)
        self.val_outputs.append({"preds": pred, "targets": label})
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.val_outputs]).cpu().numpy()
        all_targets = torch.cat([x["targets"] for x in self.val_outputs]).cpu().numpy()
        self.val_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

# Load trained model weights
model = PneumoniaModel()
checkpoint = torch.load("weights_3.ckpt", map_location=torch.device('cpu'))
state_dict = checkpoint["state_dict"]
model.load_state_dict(state_dict)
model.eval()

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

# Function to load and preprocess different file types
def load_image(file_path, file_type):
    file_type = file_type.lower()
    
    try:
        if file_type in ["png", "jpg", "jpeg"]:
            # For file objects from streamlit
            if hasattr(file_path, 'read'):
                image = Image.open(file_path).convert("L")  # Convert to grayscale
            else:
                image = Image.open(file_path).convert("L")
            image = np.array(image)
            
        elif file_type == "dcm":
            # For file objects from streamlit
            if hasattr(file_path, 'read'):
                # Create a temporary BytesIO object
                temp_file = io.BytesIO(file_path.read())
                file_path.seek(0)  # Reset pointer for future reads
                dicom_data = pydicom.dcmread(temp_file)
            else:
                dicom_data = pydicom.dcmread(file_path)
                
            image = dicom_data.pixel_array
            
        elif file_type in ["nii", "nii.gz"]:
            # For file objects from streamlit
            if hasattr(file_path, 'read'):
                # We need to save temporarily for nibabel
                with open("temp_file." + file_type, "wb") as f:
                    f.write(file_path.read())
                file_path.seek(0)  # Reset pointer for future reads
                nifti_data = nib.load("temp_file." + file_type)
                # Clean up the temp file
                import os
                try:
                    os.remove("temp_file." + file_type)
                except:
                    pass  # Ignore cleanup errors
            else:
                nifti_data = nib.load(file_path)
                
            image = nifti_data.get_fdata()
            image = np.squeeze(image)  # Only one squeeze needed
            
        else:
            return None
        
        # Common processing for all image types
        # Normalize to 0-255 range if needed
        if image.max() > 1.0 and image.max() <= 255:
            # Already in 0-255 range, no need to normalize
            pass
        else:
            # Normalize to 0-255
            image = np.uint8(255 * (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-10))  # Added small value to prevent division by zero
        
        # Resize to model's expected input size
        image = cv2.resize(image, (256, 256))
        
        # Apply the preprocessing and return tensor
        return preprocess_image(image)
        
    except Exception as e:
        import traceback
        st.error(f"Error processing image: {str(e)}")
        st.code(traceback.format_exc())
        return None

# Streamlit Web App
st.set_page_config(
    page_title="PneumoFind",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #3498db;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-normal {
        padding: 20px;
        border-radius: 10px;
        background-color: #2ecc71;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .result-pneumonia {
        padding: 20px;
        border-radius: 10px;
        background-color: #e74c3c;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
        padding: 20px;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
    .stImage img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>PneumoFind</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Advanced AI-Powered Pneumonia Detection</h2>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("steth.png")
    st.markdown("## About PneumoFind")
    st.info(
        "PneumoFind uses deep learning to analyze chest X-rays and "
        "detect signs of pneumonia with high accuracy. Upload your medical "
        "image for instant analysis."
    )
    
    st.markdown("## Supported Formats")
    st.markdown("- X-ray Images (PNG, JPG, JPEG)")
    
    
    st.markdown("## Interpretation")
    st.success("**Normal**: No signs of pneumonia detected")
    st.error("**Pneumonia**: Signs of pneumonia detected")

# Main content
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an X-ray image for analysis", type=["png", "jpg", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)

# Process image if uploaded
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Uploaded Image")
        st.image(uploaded_file, caption="X-ray Image", use_container_width=True)
    
    with col2:
        st.markdown("### Analysis Results")
        
        # Progress bar for analysis simulation
        with st.spinner("Analyzing image..."):
            # Process the image
            file_extension = uploaded_file.name.split(".")[-1]
            processed_image = load_image(uploaded_file, file_extension)
            
            if processed_image is not None:
                # Process with model
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Add a small delay for visual effect
                    progress_bar.progress(i + 1)

                # Get prediction
                with torch.no_grad():
                    output = model(processed_image)  # Model outputs raw logits
                    probability = torch.sigmoid(output).item()  # Convert logits to probability
                    prediction = "Pneumonia Detected" if probability > 0.15 else "No Pneumonia Detected"

                # Display results
                if probability > 0.15:
                    st.markdown(f"<div class='result-pneumonia'>{prediction}</div>", unsafe_allow_html=True)
                    st.warning(f"Confidence Score: {probability:.2f}")  # Display correct probability
                    st.markdown("#### Recommendation")
                    st.error("Please consult a healthcare professional for proper diagnosis and treatment.")
                else:
                    st.markdown(f"<div class='result-normal'>{prediction}</div>", unsafe_allow_html=True)
                    st.info(f"Confidence Score: {1 - probability:.2f}")  # Correct confidence display
                    st.markdown("#### Recommendation")
                    st.success("X-ray appears normal. Continue regular health monitoring.")

            else:
                st.error("Error: File format not supported or corrupted image.")
else:
    # Display sample image gallery when no file is uploaded
    st.markdown("### Sample X-rays")
    st.info("Upload an X-ray image to get started. Here are example images for reference.")
    
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        st.image("nopneumoniaxray.png", 
                caption="Example of a normal chest X-ray", width=300)
    with sample_col2:
        st.image("pneumoniaxray.png", 
                caption="Example of a pneumonia chest X-ray", width=300)

# Informational section
st.markdown("## About Pneumonia")
expander = st.expander("Learn more about pneumonia")
with expander:
    st.markdown("""
    Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, 
    causing cough with phlegm or pus, fever, chills, and difficulty breathing. Various organisms, including bacteria, 
    viruses and fungi, can cause pneumonia.
    
    **Common symptoms include:**
    - Chest pain when breathing or coughing
    - Confusion or changes in mental awareness (in adults age 65 and older)
    - Cough, which may produce phlegm
    - Fatigue
    - Fever, sweating and shaking chills
    - Lower than normal body temperature (in adults older than age 65 and people with weak immune systems)
    - Nausea, vomiting or diarrhea
    - Shortness of breath
    """)

# Footer
st.markdown("<div class='footer'>App Developed by Syed Faizan | © 2025 PneumoFind</div>", unsafe_allow_html=True)