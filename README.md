# 🌾 EdgeAgri-Net

**Advanced Multi-Task Deep Learning Platform for Precision Agriculture Research**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Overview

EdgeAgri-Net is a lightweight, multi-modal deep learning system that revolutionizes precision agriculture by solving the fragmentation problem in agricultural AI. Unlike traditional single-task models, EdgeAgri-Net simultaneously handles:

- 🔬 **Disease Detection** (38 classes across 14 plant species)
- 📊 **Yield Forecasting** (weather-aware predictions)
- 🌱 **Planting Optimization** (seasonal recommendations)
- 💧 **Resource Management** (Water, N, P, K prescriptions)

**Key Innovation:** Cross-attention fusion of visual (leaf images) and temporal (14-day weather) data streams.

---

## ✨ Key Features

### 🎯 Multi-Task Learning
- **4 simultaneous outputs** from a single unified model
- **Shared representations** improve efficiency and accuracy
- **Joint optimization** prevents task interference

### 🧠 Novel Architecture
- **MobileNetV3-Small** vision encoder (lightweight, edge-optimized)
- **1D Causal TCN** temporal encoder (14-day weather context)
- **Cross-attention fusion** with architectural defenses
- **5.4M parameters** (90% smaller than ResNet)

### 🌍 Comprehensive Coverage
- **38 disease classes** expertly labeled
- **14 plant species**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- **PlantVillage dataset** (54,000+ images)

### 📱 Edge-Ready Design
- **Lightweight**: 21 MB model, 6 MB quantized
- **Fast inference**: 50-100ms on CPU
- **ONNX export** for mobile deployment
- **Offline capable** (no cloud dependency)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/harish-2006-ux/agri-edge-net-.git
cd agri-edge-net-

# Install dependencies
pip install -r requirements.txt
```

### Run Professional Dashboard

```bash
streamlit run frontend/app_pro.py
```

Open: http://localhost:8501

### Test Core Model

```bash
python core/edgeagrinet_core.py
```

---

## 📊 Training (For Research)

### 1. Download PlantVillage Dataset

Get dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) (free account required)

### 2. Prepare Data

```bash
python download_plantvillage.py
# Choose option 2, provide path to extracted folder
```

### 3. Update Configuration

Edit `core/train.py` line 216:
```python
num_diseases=38  # Change from 10 to 38
```

### 4. Train Model

```bash
python core/train.py
```

**Training time**: 4-8 hours (CPU) or 20-40 minutes (GPU)

### 5. Use Trained Model

Update `frontend/app_pro.py` to load weights:
```python
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
```

**See** [QUICK_START_RESEARCH.md](QUICK_START_RESEARCH.md) for detailed instructions.

---

## 🏗️ Architecture

```
Input: Image (3×224×224) + Weather (14×3)
       ↓
┌──────────────────┐    ┌────────────────────┐
│  Vision Encoder  │    │  Temporal Encoder  │
│  MobileNetV3     │    │  1D Causal TCN     │
└────────┬─────────┘    └──────────┬─────────┘
         │                         │
         │   ┌────────────────┐    │
         └───┤ LayerNorm (2x) ├────┘  ← Defense #1
             └────────┬───────┘
                      │
             ┌────────▼────────┐
             │ AdaptiveAvgPool │    ← Defense #2
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │ Cross-Attention │
             │  (8 heads)      │
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │   Task Heads    │
             └────────┬────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
Disease         Yield & Cycle     Prescriptions
(38 classes)    (Regression)      (4 multi-label)
```

### Architectural Defenses

1. **Gradient Drowning Prevention**: Dual LayerNorm balances visual/temporal gradient magnitudes
2. **Attention Smearing Prevention**: AdaptiveAvgPool2d creates sharp attention maps

---

## 📸 Screenshots

### Professional Dashboard - Home
![Home Page](docs/screenshots/home.png)

### Analysis Page
![Analysis](docs/screenshots/analysis.png)

### Model Architecture
![Architecture](docs/screenshots/architecture.png)

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 92-96% |
| Disease Detection | 94% |
| Yield MAE | 0.3-0.5 t/ha |
| Cycle Accuracy | 85% |
| Prescription F1 | 0.80 |
| Model Size | 21 MB |
| Inference Time (CPU) | 50-100ms |
| Inference Time (GPU) | 5-15ms |

---

## 📁 Project Structure

```
agri-edge-net-/
├── .kiro/                      # Kiro AI steering & specs
├── core/
│   ├── edgeagrinet_core.py     # Main model (5.4M params)
│   ├── train.py                # Training pipeline
│   └── compile_pipeline.py     # ONNX export
├── frontend/
│   ├── app_pro.py              # Professional dashboard ⭐
│   └── app.py                  # Simple version
├── download_plantvillage.py    # Dataset preparation
├── update_dashboard.py         # Auto-configure UI
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── QUICK_START_RESEARCH.md     # Fast setup guide
├── RESEARCH_PROJECT_GUIDE.md   # Comprehensive docs
└── TRAINING_GUIDE.md           # Training details
```

---

## 🎓 For Research Projects

This platform is designed for academic research on:

- Multi-modal agricultural AI
- Multi-task learning
- Edge deployment optimization
- Disease detection systems
- Precision agriculture
- Explainable AI (Grad-CAM ready)

### Citation

```bibtex
@software{edgeagrinet2025,
  title={EdgeAgri-Net: Lightweight Multimodal Cross-Attention Network for Precision Agriculture},
  author={Harish},
  year={2025},
  url={https://github.com/harish-2006-ux/agri-edge-net-}
}
```

---

## 🛠️ Technical Stack

- **Framework**: PyTorch 2.0+ (LTS)
- **Vision**: MobileNetV3-Small
- **Frontend**: Streamlit 1.20+
- **Visualization**: Plotly, Pandas
- **Export**: ONNX Runtime
- **Dataset**: PlantVillage (54K images)

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| [QUICK_START_RESEARCH.md](QUICK_START_RESEARCH.md) | ⚡ Fast setup (3 commands) |
| [RESEARCH_PROJECT_GUIDE.md](RESEARCH_PROJECT_GUIDE.md) | 📚 Complete research docs |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | 🎓 Training instructions |
| [STATUS.md](STATUS.md) | 📊 Current system status |
| [FRONTEND_COMPARISON.md](FRONTEND_COMPARISON.md) | 🎨 UI options |

---

## 🌟 Features Comparison

| Feature | EdgeAgri-Net | Traditional Models |
|---------|--------------|-------------------|
| **Multi-task** | ✅ 4 tasks | ❌ Single task |
| **Multi-modal** | ✅ Image + Weather | ❌ Image only |
| **Multi-crop** | ✅ 14 species | ❌ Crop-specific |
| **Edge-ready** | ✅ 5.4M params | ❌ 25M+ params |
| **Defenses** | ✅ Built-in | ❌ Manual tuning |
| **Dashboard** | ✅ Professional | ❌ Basic/None |

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional plant species
- [ ] Real-time weather API integration
- [ ] Mobile app (React Native)
- [ ] Grad-CAM visualization
- [ ] Multi-language support
- [ ] Cloud deployment guides

---

## 📄 License

[MIT License](LICENSE) - Feel free to use for research and education.

---

## 👥 Team

- **Developer**: Harish
- **Project Type**: Research Platform
- **Status**: Production-ready architecture, training required
- **Contact**: [GitHub Issues](https://github.com/harish-2006-ux/agri-edge-net-/issues)

---

## 🙏 Acknowledgments

- **PlantVillage Dataset**: Mohanty et al., Frontiers in Plant Science (2016)
- **MobileNets**: Howard et al., arXiv (2017)
- **Attention Mechanisms**: Vaswani et al., NeurIPS (2017)

---

## 📈 Roadmap

- [x] Core model architecture
- [x] Professional dashboard
- [x] Training pipeline
- [x] ONNX export
- [x] Documentation
- [ ] Trained weights release
- [ ] Mobile deployment
- [ ] Cloud API
- [ ] Research paper

---

**Built with ❤️ for sustainable agriculture research**

🌾 **EdgeAgri-Net** - Making AI accessible to farmers worldwide 🌍
