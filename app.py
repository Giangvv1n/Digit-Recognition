import os
import sys
import subprocess
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import scipy.ndimage as ndimage
import streamlit.components.v1 as components

# Adjust sys.path to find modules in src/
sys.path.append(os.path.abspath('src'))

# Set page config for premium look
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for light theme and clean modern UI
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
        color: #334155 !important;
        font-weight: 600;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.5px;
    }
    .main-title {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subtitle {
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
    /* Card design */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    /* Glowing digit container */
    .glowing-digit-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 200px;
        border-radius: 16px;
        background: rgba(37, 99, 235, 0.05);
        border: 1px dashed rgba(37, 99, 235, 0.3);
        margin-bottom: 20px;
    }
    .glowing-digit {
        font-size: 6.5rem;
        font-weight: 800;
        color: #2563eb;
        text-shadow: 0 4px 20px rgba(37, 99, 235, 0.25);
        line-height: 1;
    }
    .digit-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 10px;
        font-weight: 600;
    }
    /* Tab headers */
    button[data-baseweb="tab"] {
        color: #64748b !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563eb !important;
        border-bottom-color: #2563eb !important;
    }
    
    /* OVERRIDE STREAMLIT WIDGETS FOR HIGH CONTRAST & PREMIUM LIGHT THEME */
    
    /* 1. Buttons (e.g., Retrain button) */
    .stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #2563eb !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.15), 0 2px 4px -1px rgba(37, 99, 235, 0.1) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: #ffffff !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3), 0 4px 6px -2px rgba(37, 99, 235, 0.15) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* 2. File Uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 16px !important;
        background-color: #f8fafc !important;
        padding: 20px !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #94a3b8 !important;
    }
    [data-testid="stFileUploader"] section button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stFileUploader"] section button:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }
    
    /* 3. Radio Buttons */
    [data-testid="stRadio"] label {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] {
        background-color: #f1f5f9 !important;
        padding: 8px !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] label * {
        color: #0f172a !important; /* Force all child elements of radio options to be dark slate */
        font-weight: 500 !important;
    }
    
    /* 4. Sliders */
    [data-testid="stSlider"] label {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSlider"] div[role="slider"] {
        background-color: #2563eb !important;
    }
    
    /* 5. Selectboxes */
    [data-testid="stSelectbox"] label {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 6. Style the canvas drawing area and toolbar container */
    div[data-testid="stCanvas"] {
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        background-color: #ffffff !important; /* Keep outer padding white */
        padding: 16px;
    }
    iframe {
        background-color: #ffffff !important; /* Set canvas background wrapper to white */
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_nn_model(path):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            st.error(f"Error loading model {path}: {e}")
    return None

@st.cache_resource
def load_svm_model(model_path, pca_path):
    if os.path.exists(model_path) and os.path.exists(pca_path):
        try:
            with open(model_path, "rb") as f:
                svm = pickle.load(f)
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
            return svm, pca
        except Exception as e:
            st.error(f"Error loading SVM model: {e}")
    return None, None

@st.cache_data
def get_sample_images_and_distribution():
    """Loads a small sample of the training dataset to visualize shapes and distributions."""
    try:
        from data_loader import load_data
        X_train, _, y_train, _, _ = load_data(data_dir="data", val_size=0.1)
        
        # Calculate label distribution
        unique, counts = np.unique(y_train, return_counts=True)
        dist_df = pd.DataFrame({
            'Digit': [str(x) for x in unique],
            'Sample Count': counts
        })
        
        # Extract one sample image for each digit class
        samples = []
        sample_labels = []
        for digit in range(10):
            idx_list = np.where(y_train == digit)[0]
            if len(idx_list) > 0:
                idx = idx_list[0]
                samples.append(X_train[idx].reshape(28, 28))
                sample_labels.append(digit)
                
        return dist_df, samples, sample_labels, True
    except Exception as e:
        # Fallback if datasets can't be imported yet
        return None, None, None, False

def run_training():
    """Runs train_compare.py as a subprocess and captures output."""
    st.info("Starting model training and comparison pipeline. This process may take a few minutes on CPU...")
    
    python_exe = sys.executable
    script_path = os.path.join("src", "train_compare.py")
    
    log_area = st.empty()
    try:
        process = subprocess.Popen(
            [python_exe, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True if os.name == 'nt' else False
        )
        
        full_logs = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                full_logs += output
                log_area.code(full_logs[-1500:]) # Show last 1500 chars of logs
                
        rc = process.poll()
        if rc == 0:
            st.success("Training completed successfully!")
            st.cache_resource.clear()
            st.cache_data.clear()
            return True
        else:
            st.error(f"Error occurred during training. Exit code: {rc}")
    except Exception as e:
        st.error(f"Could not launch training script: {e}")
    return False

def preprocess_uploaded_image(pil_img):
    """
    Processes an uploaded screenshot or image file to make it compatible with MNIST:
    - Composites transparent images onto a white background.
    - Converts to grayscale.
    - Automatically inverts background color if background is light/white.
    - Cleans background noise via thresholding.
    - Crops around the digit bounding box.
    - Resizes keeping aspect ratio so that max dimension is 20 pixels (MNIST standard).
    - Centers it inside a 28x28 black canvas.
    """
    # 1. Handle transparency (alpha channel)
    if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
        background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[-1])
        img_gray = background.convert('L')
    else:
        img_gray = pil_img.convert('L')
        
    img_arr = np.array(img_gray).astype('float32')
    
    # 2. Contrast stretching (to normalize background and text values)
    min_val = img_arr.min()
    max_val = img_arr.max()
    if max_val > min_val:
        img_arr = (img_arr - min_val) / (max_val - min_val) * 255.0
        
    # 3. Check if background is light and needs inverting
    # We inspect the mean brightness of outer border pixels
    h, w = img_arr.shape
    border_pixels = np.concatenate([
        img_arr[0, :], img_arr[-1, :], img_arr[:, 0], img_arr[:, -1]
    ])
    if np.mean(border_pixels) > 120:
        # Invert: white background (255) -> black background (0), dark text -> white text
        img_arr = 255.0 - img_arr
        
    # 4. Clean background noise (thresholding)
    img_arr = np.where(img_arr < 50, 0, img_arr).astype('uint8')
    
    # 5. Connected Component Analysis using scipy.ndimage
    # This helps isolate the digit from screenshot borders or surrounding noise
    binary_mask = (img_arr > 30).astype(int)
    labeled, num_features = ndimage.label(binary_mask)
    
    if num_features > 0:
        # Find which labels touch the border of the image
        border_mask = np.zeros_like(labeled, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        
        # Get labels that touch the border
        border_labels = np.unique(labeled[border_mask])
        
        # Clear components that touch the border (exclude background label 0)
        # We only do this if it doesn't clear the ENTIRE image (fallback protection)
        cleared_img_arr = img_arr.copy()
        cleared_binary_mask = binary_mask.copy()
        for lbl in border_labels:
            if lbl != 0:
                cleared_img_arr[labeled == lbl] = 0
                cleared_binary_mask[labeled == lbl] = 0
                
        # Check if anything is left after border clearing
        _, temp_features = ndimage.label(cleared_binary_mask)
        if temp_features > 0:
            # Safely use the border-cleared version
            img_arr = cleared_img_arr
            binary_mask = cleared_binary_mask
            labeled, num_features = ndimage.label(binary_mask)
            
    if num_features > 0:
        # Find the largest remaining connected component
        component_sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(component_sizes) + 1
        
        # Keep only the largest component (digit)
        digit_mask = (labeled == largest_label)
        img_arr = np.where(digit_mask, img_arr, 0)
        
    # 6. Crop to the isolated digit bounding box
    non_zero_coords = np.argwhere(img_arr > 30)
    if len(non_zero_coords) > 0:
        ymin, xmin = non_zero_coords.min(axis=0)
        ymax, xmax = non_zero_coords.max(axis=0)
        
        # Add a tiny padding to bounding box
        pad = int(min(h, w) * 0.02)
        ymin = max(0, ymin - pad)
        xmin = max(0, xmin - pad)
        ymax = min(h, ymax + pad)
        xmax = min(w, xmax + pad)
        
        digit_crop = img_arr[ymin:ymax, xmin:xmax]
    else:
        digit_crop = img_arr
        
    # 7. Scale keeping aspect ratio so that max dimension is 20 pixels (MNIST standard)
    crop_h, crop_w = digit_crop.shape
    if crop_h == 0 or crop_w == 0:
        return np.zeros((28, 28), dtype='float32')
        
    # Determine scaling factor
    scale = 20.0 / max(crop_h, crop_w)
    new_h = int(round(crop_h * scale))
    new_w = int(round(crop_w * scale))
    new_h = max(1, min(20, new_h))
    new_w = max(1, min(20, new_w))
    
    # Resize cropped digit
    pil_crop = Image.fromarray(digit_crop.astype('uint8'))
    # Use BILINEAR to introduce slight smoothing, matching MNIST's soft edges
    pil_resized = pil_crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
    arr_resized = np.array(pil_resized)
    
    # 8. Paste into the center of a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype='float32')
    y_start = (28 - new_h) // 2
    x_start = (28 - new_w) // 2
    canvas[y_start:y_start+new_h, x_start:x_start+new_w] = arr_resized
    
    # 9. Normalize to [0.0, 1.0]
    img_norm = canvas / 255.0
    return img_norm

def main():
    st.markdown('<h1 class="main-title">Handwritten Digit Recognizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Project: Model Comparison, Data Augmentation, Hyperparameter Tuning & Real-time Inference</p>', unsafe_allow_html=True)

    # Check for models availability
    model_paths = {
        'Improved CNN': 'models/improved_cnn.keras',
        'LeNet-5': 'models/lenet5.keras',
        'Mini-ResNet': 'models/mini_resnet.keras',
        'SVM': ('models/svm_model.pkl', 'models/pca_model.pkl')
    }
    
    models_ready = True
    for name, path in model_paths.items():
        if isinstance(path, tuple):
            if not (os.path.exists(path[0]) and os.path.exists(path[1])):
                models_ready = False
        else:
            if not os.path.exists(path):
                models_ready = False

    # Sidebar setup
    st.sidebar.markdown("### ⚙️ PREDICTION CONFIGURATION")
    selected_model_name = st.sidebar.selectbox(
        "Select prediction model:",
        ["Improved CNN", "Mini-ResNet", "LeNet-5", "SVM"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏋️ MODEL TRAINING")
    if st.sidebar.button("Retrain & Compare (All Models)"):
        with st.spinner("Running training pipeline..."):
            if run_training():
                st.rerun()

    # Create main tabs
    tab_interface, tab_dataset, tab_comparison = st.tabs([
        "🔮 Real-time Prediction", 
        "📊 Dataset Overview", 
        "📈 Performance Comparison"
    ])

    # Load selected model
    model = None
    svm_pca = (None, None)
    
    if models_ready:
        if selected_model_name == 'SVM':
            svm_pca = load_svm_model(model_paths['SVM'][0], model_paths['SVM'][1])
        else:
            model = load_nn_model(model_paths[selected_model_name])

    # TAB 1: INTERACTIVE PREDICTION INTERFACE
    with tab_interface:
        if not models_ready:
            st.warning("⚠️ Trained models not found in the 'models/' directory. Please click the 'Retrain & Compare' button in the Sidebar to train all models first.")
            
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            input_mode = st.radio("Select Input Method:", ["Draw on Canvas", "Upload Screenshot / Image File"], horizontal=True)
            
            img_norm = None
            
            if input_mode == "Draw on Canvas":
                brush_width = st.slider("Brush Width:", 10, 35, 20)
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.0)",  # Transparent fill
                    stroke_width=brush_width,
                    stroke_color="#FFFFFF",                  # White strokes
                    background_color="#000000",              # Black background
                    update_streamlit=True,
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="canvas",
                )
                
                if canvas_result.image_data is not None:
                    raw_img = canvas_result.image_data
                    if np.max(raw_img) > 0:
                        # Convert to PIL
                        img_pil = Image.fromarray(raw_img.astype('uint8')).convert('L')
                        # Resize
                        img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
                        # Normalize
                        img_norm = np.array(img_resized).astype('float32') / 255.0
            
            else:  # Upload Image Mode
                uploaded_file = st.file_uploader("Upload a digit screenshot or image file (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    pil_uploaded = Image.open(uploaded_file)
                    st.image(pil_uploaded, caption="Original Uploaded Image", width=250)
                    
                    # Preprocess
                    with st.spinner("Preprocessing image..."):
                        img_norm = preprocess_uploaded_image(pil_uploaded)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown("### Recognition Results")
            
            if img_norm is not None and models_ready:
                # Prepare batch inputs
                img_cnn = img_norm.reshape(1, 28, 28, 1)
                
                prediction_probs = None
                predicted_class = None
                
                # Inference
                if selected_model_name == 'SVM':
                    svm_model, pca_model = svm_pca
                    if svm_model is not None and pca_model is not None:
                        img_flat = img_norm.reshape(1, 784)
                        img_pca = pca_model.transform(img_flat)
                        prediction_probs = svm_model.predict_proba(img_pca)[0]
                        predicted_class = np.argmax(prediction_probs)
                else:
                    if model is not None:
                        prediction_probs = model.predict(img_cnn, verbose=0)[0]
                        predicted_class = np.argmax(prediction_probs)
                
                if prediction_probs is not None:
                    sub_col1, sub_col2 = st.columns([1, 2])
                    with sub_col1:
                        st.markdown(f"""
                        <div class="glowing-digit-container">
                            <span class="glowing-digit">{predicted_class}</span>
                            <span class="digit-label">Result</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.metric("Confidence Score", f"{prediction_probs[predicted_class]*100:.2f}%")
                        
                    with sub_col2:
                        st.write("Probability Distribution:")
                        chart_data = pd.DataFrame({
                            'Digit': [str(i) for i in range(10)],
                            'Confidence (%)': prediction_probs * 100
                        })
                        st.bar_chart(chart_data.set_index('Digit'), height=190)
                
                # Show processed input
                with st.expander("Show preprocessed input image (28x28 grayscale, MNIST-style)"):
                    st.image(img_norm, caption="Grayscale normalized 28x28 format", width=140)
            else:
                if not models_ready:
                    st.info("Models are not ready. Please launch training from the sidebar first.")
                else:
                    st.info("No input drawing or image detected. Please draw on the canvas or upload an image file.")
            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: DATASET OVERVIEW
    with tab_dataset:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 📊 Dataset Overview")
        st.markdown("""
        ### 1. Data Sources
        The primary dataset used for training is **MNIST (Modified National Institute of Standards and Technology)**, which corresponds to the **Kaggle Digit Recognizer** competition data. Additionally, the **USPS digit dataset** is utilized for cross-domain validation to test model generalization under domain shift.
        
        - **MNIST Dataset**:
          - **Training Set**: **60,000** grayscale images (or 42,000 in Kaggle CSV format) of handwritten digits.
          - **Validation/Test Set**: **10,000** grayscale images.
          - **Resolution**: $28 \times 28$ pixels.
        - **USPS Dataset**:
          - **Total Samples**: **9,298** handwritten digit images (7,291 training, 2,007 test samples).
          - **Resolution**: $16 \times 16$ pixels, scaled up to $28 \times 28$ to match MNIST format.
        """)
        
        # Load sample data inside st.spinner
        with st.spinner("Loading sample data for visualization..."):
            dist_df, samples, sample_labels, has_data = get_sample_images_and_distribution()
            
        if has_data:
            st.markdown("### 2. Visualization of Training Set Samples")
            st.write("Here are some actual digit images from the training dataset pipeline (28x28 pixel grayscale format):")
            
            # Show samples in a grid
            col_samples = st.columns(10)
            for i in range(10):
                with col_samples[i]:
                    st.image(samples[i], caption=f"Label: {sample_labels[i]}", use_container_width=True)
            
            # Show distribution chart
            st.markdown("### 3. Training Set Class Distribution")
            st.write("Below is the distribution of label counts (0-9). The dataset is balanced, which prevents model bias.")
            st.bar_chart(dist_df.set_index('Digit'))
        else:
            st.info("💡 Note: Please click 'Retrain & Compare' in the sidebar to load dataset sample images and display the distribution chart.")
            
        st.markdown("""
        ### 4. Data Preprocessing & Augmentation Pipeline
        To achieve maximum classification performance and high generalization, the data goes through the following pipeline:
        1. **Pixel Normalization**: Converts raw integer pixel values from $[0, 255]$ (or $[-1.0, 1.0]$ in USPS) to float values in the range $[0.0, 1.0]$.
        2. **Dimensionality Reshaping**:
           - **Deep Learning Models (CNN, LeNet, ResNet)**: 784-dimensional flat arrays are reshaped into 3D grids $(28, 28, 1)$ to preserve spatial layout.
           - **SVM Model**: Pixels are kept flat but reduced using PCA.
        3. **Data Augmentation**:
           To increase robustness and prevent overfitting, the training images are augmented on-the-fly using:
           - Random Rotation (up to 8%, approx $\pm 28^\circ$)
           - Random Translation (up to 8% vertically/horizontally)
           - Random Zoom (up to 8%)
        4. **Dimensionality Reduction (PCA)**:
           For the SVM model, images are projected from 784 features down to **50 principal components** using PCA. This retains $>95\%$ variance while reducing training time by **10x**.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 3: PERFORMANCE COMPARISON
    with tab_comparison:
        # Load reports images if exist
        comp_img_path = "reports/model_comparison.png"
        hist_img_path = "reports/training_histories.png"
        usps_img_path = "reports/mnist_vs_usps_comparison.png"
        report_txt_path = "reports/classification_reports.txt"
        
        if os.path.exists(comp_img_path) or os.path.exists(hist_img_path) or os.path.exists(usps_img_path):
            rep_col1, rep_col2 = st.columns(2)
            with rep_col1:
                if os.path.exists(comp_img_path):
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.image(comp_img_path, caption="Validation Accuracy vs. Training Time Comparison", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                if os.path.exists(usps_img_path):
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.image(usps_img_path, caption="Generalization Ability: MNIST (In-Domain) vs. USPS (Cross-Domain)", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            with rep_col2:
                if os.path.exists(hist_img_path):
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.image(hist_img_path, caption="Loss & Accuracy Curves Over Training Epochs", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            # Classification reports text
            if os.path.exists(report_txt_path):
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Detailed Classification Reports")
                with open(report_txt_path, "r") as f:
                    st.code(f.read(), language="text")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Confusion matrices inside tabs
            st.markdown("### 🎯 Confusion Matrices")
            tabs = st.tabs(["Improved CNN", "Mini-ResNet", "LeNet-5", "SVM"])
            models_list = ["Improved CNN", "Mini-ResNet", "LeNet-5", "SVM"]
            model_cm_filenames = ["confusion_matrix_cnn.png", "confusion_matrix_resnet.png", "confusion_matrix_lenet.png", "confusion_matrix_svm.png"]
            
            for tab, name, filename in zip(tabs, models_list, model_cm_filenames):
                with tab:
                    cm_path = os.path.join("reports", filename)
                    if os.path.exists(cm_path):
                        st.image(cm_path, caption=f"Confusion matrix for {name}", width=500)
                    else:
                        st.info(f"Confusion matrix for {name} not found. Please run training first.")
        else:
            st.info("Performance comparison data not found. Please click the training button in the left sidebar to train and generate reports.")

    # Inject Javascript to style the canvas iframe buttons (blue by default, red on hover)
    components.html("""
    <script>
        const injectCSS = () => {
            try {
                const parentDoc = window.parent.document;
                const iframes = parentDoc.querySelectorAll('iframe');
                iframes.forEach(iframe => {
                    const src = iframe.src || '';
                    const title = iframe.title || '';
                    if (
                        src.includes('streamlit_drawable_canvas') || 
                        title.includes('streamlit_drawable_canvas') ||
                        src.includes('st_canvas') ||
                        title.includes('st_canvas')
                    ) {
                        try {
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            if (iframeDoc) {
                                if (!iframeDoc.getElementById('custom-canvas-styles')) {
                                    const style = iframeDoc.createElement('style');
                                    style.id = 'custom-canvas-styles';
                                    style.innerHTML = `
                                        body {
                                            background-color: #ffffff !important;
                                        }
                                        /* Style the canvas toolbar images as high-contrast buttons */
                                        img[class*="CanvasToolbar_enabled"] {
                                            filter: invert(31%) sepia(74%) saturate(5776%) hue-rotate(219deg) brightness(95%) contrast(97%) !important; /* Blue color filter (#2563eb) */
                                            background-color: rgba(37, 99, 235, 0.08) !important;
                                            border-radius: 8px !important;
                                            padding: 6px !important;
                                            margin: 4px 6px !important;
                                            cursor: pointer !important;
                                            transition: all 0.2s ease !important;
                                            display: inline-block !important;
                                            height: 24px !important;
                                            width: 24px !important;
                                        }
                                        img[class*="CanvasToolbar_enabled"]:hover {
                                            filter: invert(38%) sepia(84%) saturate(5043%) hue-rotate(338deg) brightness(96%) contrast(97%) !important; /* Red color filter (#ef4444) */
                                            background-color: rgba(239, 68, 68, 0.15) !important;
                                            transform: scale(1.15) !important;
                                        }
                                        img[class*="CanvasToolbar_disabled"] {
                                            filter: grayscale(1) !important;
                                            opacity: 0.25 !important;
                                            background-color: transparent !important;
                                            border-radius: 8px !important;
                                            padding: 6px !important;
                                            margin: 4px 6px !important;
                                            cursor: not-allowed !important;
                                            display: inline-block !important;
                                            height: 24px !important;
                                            width: 24px !important;
                                        }
                                    `;
                                    iframeDoc.head.appendChild(style);
                                }
                            }
                        } catch (err) {
                            // Suppress errors
                        }
                    }
                });
            } catch (globalErr) {
                // Suppress errors
            }
        };
        setInterval(injectCSS, 1000);
    </script>
    """, height=0, width=0)

if __name__ == "__main__":
    main()
