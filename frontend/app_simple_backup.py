"""
EdgeAgri-Net Streamlit Dashboard
=================================
Interactive frontend for precision agriculture AI predictions.

Features:
- Image upload for leaf disease diagnosis
- 14-day weather input sliders
- Multi-task predictions (disease, yield, cycle, prescriptions)
- Market price simulation with scenario projections
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys

# Add core module to path
sys.path.append(str(Path(__file__).parent.parent))

from core.edgeagrinet_core import EdgeAgriNet, market_simulation


# Page configuration
st.set_page_config(
    page_title="EdgeAgri-Net Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the EdgeAgri-Net model (cached)."""
    model = EdgeAgriNet(
        num_diseases=10,
        num_cycles=4,
        num_resources=4
    )
    model.eval()
    return model


def preprocess_image(uploaded_file):
    """Preprocess uploaded image for model input."""
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((224, 224))
    
    # Convert to tensor and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    # Basic normalization (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, image


def create_weather_input():
    """Create 14-day weather input interface."""
    st.sidebar.header("📅 14-Day Weather Data")
    
    weather_data = []
    
    # Option 1: Quick fill with defaults
    if st.sidebar.button("🔄 Fill with Sample Data"):
        st.session_state.use_sample = True
    
    if st.sidebar.checkbox("Show detailed daily inputs", value=False):
        # Detailed per-day inputs
        for day in range(1, 15):
            with st.sidebar.expander(f"Day {day}"):
                temp = st.slider(f"Temperature (°C)", -10.0, 50.0, 25.0, key=f"temp_{day}")
                humidity = st.slider(f"Humidity (%)", 0.0, 100.0, 60.0, key=f"hum_{day}")
                rain = st.slider(f"Rainfall (mm)", 0.0, 200.0, 10.0, key=f"rain_{day}")
                weather_data.append([temp, humidity, rain])
    else:
        # Simplified: average values for all 14 days
        st.sidebar.subheader("Average Weather Conditions")
        avg_temp = st.sidebar.slider("Avg Temperature (°C)", -10.0, 50.0, 25.0)
        avg_humidity = st.sidebar.slider("Avg Humidity (%)", 0.0, 100.0, 60.0)
        avg_rain = st.sidebar.slider("Avg Rainfall (mm)", 0.0, 200.0, 10.0)
        
        weather_data = [[avg_temp, avg_humidity, avg_rain]] * 14
    
    # Convert to tensor
    weather_tensor = torch.tensor(weather_data, dtype=torch.float32).unsqueeze(0)
    return weather_tensor, weather_data


def display_predictions(predictions, weather_data):
    """Display multi-task prediction results."""
    
    # Get crop and disease labels from session state
    crop_type = st.session_state.get('crop_type', 'Rice')
    
    # Crop-specific disease labels
    disease_db = {
        'Rice': [
            "Healthy", "Bacterial Blight", "Brown Spot", "Leaf Smut",
            "Blast", "Tungro", "Bacterial Leaf Streak", "Sheath Rot",
            "False Smut", "Downy Mildew"
        ],
        'Tomato': [
            "Healthy", "Early Blight", "Late Blight", "Leaf Mold",
            "Septoria Leaf Spot", "Bacterial Spot", "Target Spot",
            "Mosaic Virus", "Yellow Leaf Curl Virus", "Powdery Mildew"
        ],
        'Potato': [
            "Healthy", "Early Blight", "Late Blight", "Common Scab",
            "Black Leg", "Fusarium Dry Rot", "Pink Rot", "Verticillium Wilt",
            "Rhizoctonia", "Virus"
        ],
        'Wheat': [
            "Healthy", "Leaf Rust", "Stem Rust", "Stripe Rust",
            "Powdery Mildew", "Septoria Leaf Blotch", "Tan Spot",
            "Fusarium Head Blight", "Loose Smut", "Barley Yellow Dwarf"
        ],
        'Corn': [
            "Healthy", "Northern Leaf Blight", "Common Rust", "Gray Leaf Spot",
            "Southern Leaf Blight", "Cercospora Leaf Spot", "Anthracnose",
            "Goss's Wilt", "Stewart's Wilt", "Corn Smut"
        ]
    }
    
    disease_labels = disease_db.get(crop_type, disease_db['Rice'])
    
    cycle_labels = ["Spring", "Summer", "Fall", "Winter"]
    resource_labels = ["Water", "Nitrogen", "Phosphorus", "Potassium"]
    
    # Header
    st.markdown('<div class="main-header">🌾 EdgeAgri-Net Predictions</div>', unsafe_allow_html=True)
    
    # ⚠️ UNTRAINED MODEL WARNING
    st.warning(f"""
    ⚠️ **IMPORTANT: This model is NOT TRAINED yet!**
    
    The model currently has random weights and **cannot accurately detect diseases**.
    Predictions shown are **random** and for demonstration purposes only.
    
    **Current Crop Type:** {st.session_state.get('crop_type', 'Rice')}
    
    To get real predictions, you need to:
    1. Collect labeled training data (crop images + disease labels)
    2. Train the model using the training loop
    3. Save and load the trained weights
    """)
    
    # Create 4 columns for main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # 1. Disease Diagnosis
    with col1:
        disease_idx = predictions['disease'].item()
        disease_prob = predictions['disease_probs'][0, disease_idx].item()
        
        st.markdown("### 🔬 Disease Diagnosis")
        st.metric(
            label="Detected Condition",
            value=disease_labels[disease_idx],
            delta=f"{disease_prob*100:.1f}% confidence"
        )
        
        # Top 3 probabilities
        top3_probs, top3_idx = torch.topk(predictions['disease_probs'][0], 3)
        with st.expander("View top 3 predictions"):
            for prob, idx in zip(top3_probs, top3_idx):
                st.write(f"• {disease_labels[idx]}: {prob.item()*100:.1f}%")
    
    # 2. Yield Forecast
    with col2:
        yield_value = predictions['yield_tons_per_hectare'].item()
        
        st.markdown("### 📊 Yield Forecast")
        st.metric(
            label="Expected Yield",
            value=f"{yield_value:.2f}",
            delta="tons/hectare"
        )
    
    # 3. Planting Cycle
    with col3:
        cycle_idx = predictions['cycle'].item()
        cycle_prob = predictions['cycle_probs'][0, cycle_idx].item()
        
        st.markdown("### 🌱 Optimal Cycle")
        st.metric(
            label="Best Planting Window",
            value=cycle_labels[cycle_idx],
            delta=f"{cycle_prob*100:.1f}% confidence"
        )
    
    # 4. Resource Prescription
    with col4:
        prescriptions = predictions['prescription'][0]
        active_resources = [resource_labels[i] for i, val in enumerate(prescriptions) if val]
        
        st.markdown("### 💧 Prescriptions")
        if active_resources:
            st.success(f"Adjust: {', '.join(active_resources)}")
        else:
            st.info("No adjustments needed")
    
    st.divider()
    
    # Detailed visualizations
    col_left, col_right = st.columns(2)
    
    # Disease probability chart
    with col_left:
        st.subheader("Disease Probability Distribution")
        probs = predictions['disease_probs'][0].detach().numpy()
        
        fig = go.Figure(data=[
            go.Bar(
                x=disease_labels,
                y=probs,
                marker_color=['#2E7D32' if i == disease_idx else '#BDBDBD' 
                             for i in range(len(disease_labels))]
            )
        ])
        fig.update_layout(
            xaxis_title="Disease Type",
            yaxis_title="Probability",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Weather summary
    with col_right:
        st.subheader("14-Day Weather Summary")
        
        weather_df = pd.DataFrame(
            weather_data,
            columns=["Temperature (°C)", "Humidity (%)", "Rainfall (mm)"]
        )
        weather_df.index = [f"Day {i+1}" for i in range(14)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=weather_df["Temperature (°C)"],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='#FF6B6B')
        ))
        fig.add_trace(go.Scatter(
            y=weather_df["Humidity (%)"],
            mode='lines+markers',
            name='Humidity',
            line=dict(color='#4ECDC4'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            xaxis_title="Day",
            yaxis_title="Temperature (°C)",
            yaxis2=dict(
                title="Humidity (%)",
                overlaying='y',
                side='right'
            ),
            height=400,
            legend=dict(x=0.7, y=1.1, orientation='h')
        )
        st.plotly_chart(fig, use_container_width=True)


def market_simulator(yield_pred):
    """Market price elasticity simulation interface."""
    st.divider()
    st.header("💰 Market Price Elasticity Simulator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Simulate revenue scenarios based on market price fluctuations.
        Adjust the price change percentage to see how your potential revenue changes.
        """)
        
        # Input parameters
        current_price = st.number_input(
            "Current Market Price ($/ton)",
            min_value=100.0,
            max_value=1000.0,
            value=350.0,
            step=10.0
        )
        
        production_cost = st.number_input(
            "Production Cost ($)",
            min_value=0.0,
            max_value=5000.0,
            value=800.0,
            step=50.0
        )
        
        price_change = st.slider(
            "Price Change Scenario (%)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=5.0
        )
    
    with col2:
        # Calculate scenario
        revenue = market_simulation(
            yield_hat=yield_pred,
            price_current=current_price,
            delta_p=price_change,
            production_cost=production_cost
        )
        
        st.metric(
            label="Simulated Revenue",
            value=f"${revenue:.2f}",
            delta=f"{price_change:+.0f}% price change"
        )
        
        # Quick scenario cards
        st.markdown("#### Quick Scenarios")
        
        bearish = market_simulation(yield_pred, current_price, -20, production_cost)
        baseline = market_simulation(yield_pred, current_price, 0, production_cost)
        bullish = market_simulation(yield_pred, current_price, +20, production_cost)
        
        st.markdown(f"**🐻 Bearish (-20%):** ${bearish:.2f}")
        st.markdown(f"**➖ Baseline (0%):** ${baseline:.2f}")
        st.markdown(f"**🐂 Bullish (+20%):** ${bullish:.2f}")
    
    # Revenue sensitivity chart
    st.subheader("Revenue Sensitivity Analysis")
    
    price_changes = np.linspace(-30, 30, 13)
    revenues = [
        market_simulation(yield_pred, current_price, pc, production_cost)
        for pc in price_changes
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_changes,
        y=revenues,
        mode='lines+markers',
        line=dict(color='#2E7D32', width=3),
        marker=dict(size=8)
    ))
    
    # Add break-even line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Break-even"
    )
    
    fig.update_layout(
        xaxis_title="Price Change (%)",
        yaxis_title="Revenue ($)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🌾 EdgeAgri-Net</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Lightweight Multimodal Cross-Attention Network for Precision Agriculture</div>',
        unsafe_allow_html=True
    )
    
    # Load model
    with st.spinner("Loading EdgeAgri-Net model..."):
        model = load_model()
    
    # Sidebar: Image upload
    st.sidebar.header("📸 Upload Leaf Image")
    
    # Crop selection
    crop_type = st.sidebar.selectbox(
        "Select Crop Type",
        ["Rice", "Tomato", "Potato", "Wheat", "Corn"],
        help="Choose the crop type for disease classification"
    )
    st.session_state['crop_type'] = crop_type
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a crop leaf for disease diagnosis"
    )
    
    # Weather input
    weather_tensor, weather_data = create_weather_input()
    
    # Main content
    if uploaded_file is not None:
        # Preprocess image
        img_tensor, original_image = preprocess_image(uploaded_file)
        
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(original_image, caption="Uploaded Leaf Image", use_container_width=True)
        
        st.divider()
        
        # Run inference
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing crop conditions..."):
                predictions = model.predict(img_tensor, weather_tensor)
            
            # Display results
            display_predictions(predictions, weather_data)
            
            # Market simulator
            yield_pred = predictions['yield_tons_per_hectare'].item()
            market_simulator(yield_pred)
    
    else:
        # Landing page
        st.warning("""
        ⚠️ **MODEL STATUS: UNTRAINED**
        
        This EdgeAgri-Net model is **not trained** and will produce **random predictions**.
        
        The architecture is complete, but the model needs training data to learn disease patterns.
        """)
        
        st.info("👈 Upload a leaf image and configure weather data to see the demo (predictions will be random)")
        
        # Feature showcase
        st.header("🎯 Platform Features")
        
        feat_col1, feat_col2 = st.columns(2)
        
        with feat_col1:
            st.markdown("""
            ### 🔬 Multimodal Diagnostics
            - Real-time disease detection from leaf images
            - 14-day weather context integration
            - Sharp Grad-CAM attention heatmaps
            
            ### 📊 Spatiotemporal Yield Tracking
            - Accurate harvest volume estimation
            - Image + soil moisture fusion
            - Logistics planning support
            """)
        
        with feat_col2:
            st.markdown("""
            ### 💰 Market Price Simulator
            - Live wholesale price integration
            - Bearish/Baseline/Bullish scenarios
            - Revenue sensitivity analysis
            
            ### 💧 Prescriptive Agronomist Engine
            - Quantified resource recommendations
            - Water, N, P, K optimization
            - Cost-saving action alerts
            """)
        
        # Technical specs
        with st.expander("🛠️ Technical Specifications"):
            st.markdown("""
            **Architecture:**
            - Vision Backbone: MobileNetV3-Small
            - Temporal Encoder: 1D Causal TCN
            - Fusion: Cross-Attention with LayerNorm defenses
            
            **Defenses:**
            - Defense 1: Dual LayerNorm prevents gradient drowning
            - Defense 2: AdaptiveAvgPool2d prevents attention smearing
            
            **Outputs:**
            - Disease classification (10 classes)
            - Yield regression (tons/hectare)
            - Cycle selection (4 seasons)
            - Resource prescription (4 multi-label)
            
            **Model Size:** ~5.4M parameters
            **Framework:** PyTorch LTS + ONNX export ready
            """)


if __name__ == "__main__":
    main()
