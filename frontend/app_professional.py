"""
EdgeAgri-Net Professional Research Dashboard
============================================
Advanced multi-section interface for agricultural AI research and deployment.

Features:
- Multi-page navigation (Home, Analysis, Model Info, Research, About)
- Professional visualizations with Plotly
- Detailed model architecture display
- Comparative analysis tools
- Export capabilities
- Research-grade presentation
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import sys
import io
import time

# Add core module to path
sys.path.append(str(Path(__file__).parent.parent))

from core.edgeagrinet_core import EdgeAgriNet, market_simulation


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EdgeAgri-Net | Advanced Agricultural AI Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/edgeagrinet',
        'Report a bug': 'https://github.com/yourusername/edgeagrinet/issues',
        'About': 'EdgeAgri-Net: Multi-Task Deep Learning for Precision Agriculture'
    }
)


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_custom_css():
    """Load professional CSS styling."""
    st.markdown("""
    <style>
        /* Main color scheme - Professional green theme */
        :root {
            --primary-color: #2E7D32;
            --secondary-color: #66BB6A;
            --accent-color: #FFA726;
            --dark-bg: #1E1E1E;
            --light-bg: #F5F5F5;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .main-header p {
            font-size: 1.2rem;
            margin-top: 0.5rem;
            opacity: 0.95;
        }
        
        /* Section cards */
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #2E7D32;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
        }
        
        .info-card h3 {
            color: #2E7D32;
            margin-top: 0;
        }
        
        .metric-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .metric-box h2 {
            color: #2E7D32;
            font-size: 2.5rem;
            margin: 0;
        }
        
        .metric-box p {
            color: #555;
            font-size: 1rem;
            margin-top: 0.5rem;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            margin: 0.5rem;
        }
        
        .badge-success {
            background-color: #4CAF50;
            color: white;
        }
        
        .badge-warning {
            background-color: #FF9800;
            color: white;
        }
        
        .badge-info {
            background-color: #2196F3;
            color: white;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            padding: 10px 20px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #2E7D32;
            color: white;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f5f5 0%, #e8e8e8 100%);
        }
    </style>
    """, unsafe_allow_html=True)



# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load EdgeAgri-Net model with caching."""
    try:
        model = EdgeAgriNet(num_diseases=38, num_cycles=4, num_resources=4)
        
        # Try to load trained weights
        if Path('best_model.pth').exists():
            checkpoint = torch.load('best_model.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            st.session_state['model_trained'] = True
            st.session_state['train_epoch'] = checkpoint.get('epoch', 'Unknown')
            st.session_state['train_loss'] = checkpoint.get('val_loss', 'Unknown')
        else:
            st.session_state['model_trained'] = False
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ============================================================================
# DISEASE DATABASE (PlantVillage)
# ============================================================================

PLANTVILLAGE_DISEASES = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust",
    "Apple - Healthy", "Blueberry - Healthy", "Cherry - Powdery Mildew",
    "Cherry - Healthy", "Corn - Cercospora Leaf Spot (Gray Leaf Spot)",
    "Corn - Common Rust", "Corn - Northern Leaf Blight", "Corn - Healthy",
    "Grape - Black Rot", "Grape - Esca (Black Measles)", 
    "Grape - Leaf Blight (Isariopsis Leaf Spot)", "Grape - Healthy",
    "Orange - Haunglongbing (Citrus Greening)", "Peach - Bacterial Spot",
    "Peach - Healthy", "Pepper - Bacterial Spot", "Pepper - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Raspberry - Healthy", "Soybean - Healthy", "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch", "Strawberry - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)", 
    "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus", "Tomato - Healthy"
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_image(uploaded_file):
    """Preprocess uploaded image for model input."""
    image = Image.open(uploaded_file).convert('RGB')
    original_size = image.size
    image = image.resize((224, 224))
    
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, image, original_size


def create_weather_tensor(weather_data):
    """Create weather tensor from input data."""
    weather_tensor = torch.tensor(weather_data, dtype=torch.float32).unsqueeze(0)
    return weather_tensor



# ============================================================================
# PAGE: HOME / DASHBOARD
# ============================================================================

def page_home():
    """Main dashboard with system overview."""
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 EdgeAgri-Net</h1>
        <p>Advanced Multi-Task Deep Learning Platform for Precision Agriculture</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_status = "Trained" if st.session_state.get('model_trained', False) else "Untrained"
        status_color = "🟢" if st.session_state.get('model_trained', False) else "🟡"
        st.markdown(f"""
        <div class="metric-box">
            <h2>{status_color}</h2>
            <p><strong>Model Status</strong></p>
            <p>{model_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h2>38</h2>
            <p><strong>Disease Classes</strong></p>
            <p>14 Plant Species</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h2>5.4M</h2>
            <p><strong>Parameters</strong></p>
            <p>Lightweight Architecture</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <h2>4</h2>
            <p><strong>Task Outputs</strong></p>
            <p>Multi-Task Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown("## 🎯 Platform Capabilities")
    
    feat_col1, feat_col2 = st.columns(2)
    
    with feat_col1:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 Disease Detection</h3>
            <ul>
                <li><strong>38 disease classes</strong> across 14 plant species</li>
                <li><strong>Multi-modal fusion</strong> of visual + temporal data</li>
                <li><strong>Confidence scores</strong> and probability distributions</li>
                <li><strong>Attention visualization</strong> (Grad-CAM ready)</li>
            </ul>
        </div>
        
        <div class="info-card">
            <h3>📊 Yield Forecasting</h3>
            <ul>
                <li><strong>Continuous prediction</strong> (tons/hectare)</li>
                <li><strong>Weather-aware estimation</strong> (14-day context)</li>
                <li><strong>Robust to outliers</strong> (Huber loss training)</li>
                <li><strong>Historical comparison</strong> tools</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="info-card">
            <h3>🌱 Planting Optimization</h3>
            <ul>
                <li><strong>Cycle recommendation</strong> (Spring/Summer/Fall/Winter)</li>
                <li><strong>Climate-aware suggestions</strong></li>
                <li><strong>Species-specific advice</strong></li>
                <li><strong>Growing degree days</strong> (GDD) tracking</li>
            </ul>
        </div>
        
        <div class="info-card">
            <h3>💧 Resource Prescriptions</h3>
            <ul>
                <li><strong>Multi-label classification</strong> (Water, N, P, K)</li>
                <li><strong>Actionable recommendations</strong></li>
                <li><strong>Cost-benefit analysis</strong> ready</li>
                <li><strong>Environmental impact</strong> considerations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Highlights
    st.markdown("## 🏗️ Architecture Highlights")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **Vision Encoder**
        - MobileNetV3-Small
        - 576 → 512 dim projection
        - Pre-trained on ImageNet
        - Edge-optimized design
        """)
    
    with tech_col2:
        st.markdown("""
        **Temporal Encoder**
        - 1D Causal TCN
        - 3 → 512 dim embedding
        - 14-day weather context
        - No future data leakage
        """)
    
    with tech_col3:
        st.markdown("""
        **Fusion Layer**
        - Cross-attention mechanism
        - Dual LayerNorm defenses
        - AdaptiveAvgPool2d
        - Multi-head attention (8 heads)
        """)
    
    # Model Status Warning
    if not st.session_state.get('model_trained', False):
        st.warning("""
        ⚠️ **Model Status: UNTRAINED**
        
        The model currently has random weights. Predictions shown are for demonstration only.
        
        To enable real disease detection:
        1. Download PlantVillage dataset
        2. Run `python download_plantvillage.py`
        3. Train with `python core/train.py`
        
        See **QUICK_START_RESEARCH.md** for detailed instructions.
        """)
