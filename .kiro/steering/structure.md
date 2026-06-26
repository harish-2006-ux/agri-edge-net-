# EdgeAgriNet Structure

Maps file paths & rules for Kiro's code generator

# Project Structure Rules

- All PyTorch models and model conversion tools go in the `core/` folder.
- The user interface dashboard code belongs exclusively in `frontend/app.py`.
- No deep learning logic or backpropagation loops should ever be written inside the frontend layer.
- Frontend queries `core/edgeagrinet_core.py` directly for predictions.

# File Inventory

```
EdgeAgriNet-Workspace/
├── .kiro/                           # Kiro's Core AI Context Directory
│   ├── steering/                    # Global code & architecture rules
│   │   ├── product.md               # App mission statement (Socio-Economic Agritech)
│   │   ├── tech.md                  # STRICT MATH: Formulas, LayerNorm, Defenses
│   │   └── structure.md             # Maps file paths & rules for Kiro's code generator
│   └── specs/                       # Active development task-tracks (Kiro Features)
│       └── implement_frontend/
│           ├── requirements.md      # EARS user stories (Sliders, Uploaders, Forecasts)
│           ├── design.md            # Component data flow diagram
│           └── tasks.md             # Step-by-step task checklist executed by Kiro
├── core/                            # The Heavy Deep Learning Infrastructure
│   ├── __init__.py
│   ├── edgeagrinet_core.py          # Multi-task PyTorch model script (compilable)
│   └── compile_pipeline.py          # Tracing script to export model to ONNX format
├── frontend/                        # The Interactive Client Application Layer
│   ├── __init__.py
│   └── app.py                       # Streamlit Dashboard (Sliders, Scraping, Charts)
├── requirements.txt                 # Standard Python project dependencies
└── README.md                        # General setup & onboarding documentation
```