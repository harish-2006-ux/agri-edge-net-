# EdgeAgri-Net 🌾

**Lightweight Multimodal Cross-Attention Network for Precision Agriculture**

An intelligent, multi-task deep learning platform designed to revolutionize precision agriculture on rural edge-hardware.

---

## 🎯 Overview

EdgeAgri-Net solves the fragmentation problem in agricultural AI by introducing a unified multimodal architecture that:

- **Ingests multiple data streams**: High-density spatial crop imagery + sparse 14-day weather metrics
- **Generates 4 parallel real-time outputs**: Disease diagnoses, yield forecasts, planting cycle selections, resource prescriptions
- **Optimized for edge deployment**: Lightweight MobileNetV3 backbone, ~5.4M parameters, ONNX-exportable
- **Includes economic simulation**: Dynamic market-price elasticity engine for revenue forecasting

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch LTS |
| Vision Backbone | MobileNetV3-Small |
| Temporal Encoder | 1D Causal TCN |
| Model Export | ONNX |
| Optimization | INT8 Post-Training Quantization (PTQ) |
| Frontend | Streamlit Dashboard |

---

## 📐 Core Architecture

### Multimodal Cross-Attention Fusion

Instead of simple concatenation, weather embeddings (Queries) actively attend to visual feature maps (Keys, Values):

```
Attention(Q, K, V) = Softmax(QK^T / sqrt(d_embed)) · V
```

### Built-In Architectural Defenses

1. **Defense Against Gradient Drowning**: Dual LayerNorm on visual + temporal streams before fusion
2. **Defense Against Attention Smearing**: AdaptiveAvgPool2d((1,1)) on spatial features before cross-attention

### Multi-Task Loss

```
L_total = α·L_disease + β·L_yield + γ·L_cycles + δ·L_prescription
```

- **L_disease**: Cross-Entropy for pathology classification
- **L_yield**: Huber Loss for robust yield regression
- **L_cycles**: Cross-Entropy for planting window selection
- **L_prescription**: Binary Cross-Entropy for multi-label resource recommendations

### Market Simulation

```
R_sim(ΔP) = [Y_hat × (P_current × (1 + ΔP/100))] - C_production
```

---

## 🚀 Quick Start

### ⚠️ IMPORTANT: Model Status

**The model is NOT TRAINED yet.** It has random weights and will produce random predictions. You must train it on labeled data first.

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for complete training instructions.

### 1. Installation

```bash
# Clone or navigate to the repository
cd EdgeAgriNet-Workspace

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard (Demo Mode)

```bash
streamlit run frontend/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Note:** Predictions will be random until you train the model.

### 3. Train the Model (Required for Real Use)

```bash
python core/edgeagrinet_core.py
```

Expected output:
```
EdgeAgri-Net Output Shapes:
  disease_logits: torch.Size([2, 10])
  yield_pred:     torch.Size([2, 1])
  cycle_logits:   torch.Size([2, 4])
  prescription:   torch.Size([2, 4])

Market Simulation (10% price increase):
  Revenue: $817.00

Total Parameters: 5,401,395
```

---

## 📂 Project Structure

```
EdgeAgriNet-Workspace/
├── .kiro/                        # Kiro's AI context directory
│   ├── steering/                 # Global code & architecture rules
│   │   ├── product.md            # Product mission statement
│   │   ├── tech.md               # Technical specifications
│   │   └── structure.md          # File structure & rules
│   └── specs/                    # Development task-tracks
│       └── implement_frontend/
│           ├── requirements.md   # User stories
│           ├── design.md         # Component data flow
│           └── tasks.md          # Task checklist
├── core/                         # Deep Learning Infrastructure
│   ├── __init__.py
│   ├── edgeagrinet_core.py       # Multi-task PyTorch model
│   └── compile_pipeline.py       # ONNX export script (TBD)
├── frontend/                     # Interactive Client Layer
│   ├── __init__.py
│   └── app.py                    # Streamlit dashboard
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🎨 Dashboard Features

### 1. Multimodal Diagnostics
- Upload leaf camera images
- Input 14-day weather data (temperature, humidity, rainfall)
- Get disease classification with confidence scores
- View top-3 predictions

### 2. Spatiotemporal Yield Tracking
- Predict harvest volume (tons/hectare)
- Fused image + weather context
- Support logistics planning

### 3. Market Price Simulator
- Input current market prices and production costs
- Simulate revenue under price fluctuations (-30% to +30%)
- View Bearish/Baseline/Bullish scenarios
- Interactive sensitivity analysis chart

### 4. Prescriptive Agronomist Engine
- Get actionable resource recommendations
- Water, Nitrogen, Phosphorus, Potassium adjustments
- Cost-saving alerts

---

## 🧪 Model Details

### Inputs
- **Image**: RGB leaf photo, 224×224 pixels
- **Weather**: 14-day time series, 3 features (temperature, humidity, rainfall)

### Outputs
- **Disease**: Classification logits (10 classes)
- **Yield**: Continuous prediction (tons/hectare)
- **Cycle**: Best planting window (4 seasons)
- **Prescription**: Multi-label binary flags (4 resources)

### Training
- Multi-task loss with task-specific loss functions
- Weighted sum: α=β=γ=δ=1.0 (tunable)
- Optimizer: AdamW with learning rate scheduling
- Defense mechanisms ensure balanced gradient flow

---

## 🔬 Research Contributions

1. **Multimodal Fusion**: Cross-attention between sparse temporal and dense spatial modalities
2. **Architectural Defenses**: LayerNorm + AdaptiveAvgPool2d prevent training pathologies
3. **Economic Integration**: Market simulation bridges biological insights to financial decisions
4. **Edge Optimization**: Lightweight architecture suitable for rural deployment

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Parameters | 5,401,395 |
| Input Size | Image: 224×224, Weather: 14×3 |
| Inference Speed | TBD (target: <100ms on mobile CPU) |
| Memory Footprint | TBD (target: <50MB after quantization) |

---

## 🛡️ Security & Safety

- No pretrained weights downloaded by default (network independence)
- All inference runs locally (no data sent to external servers)
- Suitable for offline/rural deployment

---

## 🔮 Future Enhancements

- [ ] ONNX export pipeline (`compile_pipeline.py`)
- [ ] INT8 quantization for mobile deployment
- [ ] Grad-CAM visualization for disease localization
- [ ] Real-time market price scraping integration
- [ ] Multi-crop support (rice, wheat, corn, etc.)
- [ ] Mobile app (iOS/Android)

---

## 📚 Citation

If you use EdgeAgri-Net in your research, please cite:

```
@software{edgeagrinet2025,
  title={EdgeAgri-Net: Lightweight Multimodal Cross-Attention Network for Precision Agriculture},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/edgeagrinet}
}
```

---

## 📄 License

[Specify your license here]

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📧 Contact

For questions or collaboration:
- Email: [your.email@example.com]
- Project Link: [https://github.com/yourusername/edgeagrinet]

---

**Built with ❤️ for sustainable agriculture**
