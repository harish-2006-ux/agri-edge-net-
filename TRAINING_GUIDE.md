# EdgeAgri-Net Training Guide 🎓

## ⚠️ IMPORTANT: The Model is NOT Trained

The current model has **random weights** and **cannot detect diseases accurately**. You need to train it on labeled agricultural data first.

---

## 📊 What You Need

### 1. Training Data

Collect labeled agricultural data:

- **Leaf Images**: Clear photos of crop leaves (healthy and diseased)
- **Weather Data**: 14-day weather records (temperature, humidity, rainfall)
- **Labels**:
  - Disease type (0-9, where 0 = healthy)
  - Yield estimate (tons/hectare)
  - Best planting cycle (0-3 for Spring/Summer/Fall/Winter)
  - Resource needs (binary flags for Water, Nitrogen, Phosphorus, Potassium)

### 2. Data Structure

Organize your data like this:

```
EdgeAgriNet-Workspace/
└── data/
    ├── train/
    │   ├── images/
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   ├── weather.csv
    │   └── labels.csv
    └── val/
        ├── images/
        ├── weather.csv
        └── labels.csv
```

---

## 📝 Data File Formats

### labels.csv

| Column | Type | Description |
|--------|------|-------------|
| image_name | string | Filename (e.g., img001.jpg) |
| disease_label | int (0-9) | Disease class (0=Healthy, 1-9=diseases) |
| yield_value | float | Expected yield (tons/hectare) |
| cycle_label | int (0-3) | Best planting cycle (0=Spring, 1=Summer, 2=Fall, 3=Winter) |
| water | int (0 or 1) | Needs water adjustment? |
| nitrogen | int (0 or 1) | Needs nitrogen adjustment? |
| phosphorus | int (0 or 1) | Needs phosphorus adjustment? |
| potassium | int (0 or 1) | Needs potassium adjustment? |

**Example:**
```csv
image_name,disease_label,yield_value,cycle_label,water,nitrogen,phosphorus,potassium
img001.jpg,2,4.5,1,1,0,1,0
img002.jpg,0,5.2,2,0,0,0,0
img003.jpg,5,3.8,1,1,1,0,1
```

### weather.csv

| Column | Type | Description |
|--------|------|-------------|
| image_name | string | Filename |
| day | int (1-14) | Day number |
| temperature | float | Temperature (°C) |
| humidity | float | Humidity (%) |
| rainfall | float | Rainfall (mm) |

**Example:**
```csv
image_name,day,temperature,humidity,rainfall
img001.jpg,1,28.5,65.0,5.2
img001.jpg,2,29.0,62.0,0.0
img001.jpg,3,27.8,68.0,12.5
...
img001.jpg,14,30.2,60.0,0.0
img002.jpg,1,25.0,70.0,8.0
...
```

---

## 🏋️ Training the Model

### Step 1: Prepare Your Data

1. Collect and label your agricultural images
2. Record 14-day weather history for each image
3. Create `labels.csv` and `weather.csv` files
4. Organize into `data/train/` and `data/val/` folders

### Step 2: Customize Disease Labels

Edit `core/train.py` line with your disease count:

```python
model = EdgeAgriNet(
    num_diseases=10,  # Change this to match your dataset
    num_cycles=4,
    num_resources=4
)
```

### Step 3: Run Training

```bash
cd EdgeAgriNet-Workspace
python core/train.py
```

The script will:
- Load your training and validation data
- Train for 50 epochs
- Save the best model to `best_model.pth`
- Show progress and losses for each task

### Step 4: Load Trained Model in Dashboard

Update `frontend/app.py` to load trained weights:

```python
@st.cache_resource
def load_model():
    """Load the EdgeAgri-Net model (cached)."""
    model = EdgeAgriNet(
        num_diseases=10,
        num_cycles=4,
        num_resources=4
    )
    
    # Load trained weights
    checkpoint = torch.load('best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model
```

---

## 🎯 Where to Get Training Data

### Option 1: Public Datasets

- **PlantVillage Dataset**: 50,000+ images, 14 crop species
  - https://www.kaggle.com/datasets/emmarex/plantdisease
  
- **Rice Leaf Disease Dataset**: 3,355 images
  - https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases

- **Tomato Disease Dataset**: 10,000+ images
  - https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources

### Option 2: Collect Your Own

1. Take clear photos of crop leaves (healthy and diseased)
2. Record weather data from local weather stations
3. Label diseases with help from agricultural experts
4. Estimate yields based on harvest records

---

## 📈 Expected Training Results

After training on a good dataset, you should see:

- **Disease Classification Accuracy**: 85-95%
- **Yield Prediction MAE**: 0.3-0.5 tons/hectare
- **Cycle Classification Accuracy**: 75-85%
- **Prescription Precision**: 70-80%

Training time varies:
- CPU: 2-6 hours per epoch (depending on dataset size)
- GPU: 10-30 minutes per epoch

---

## 🔧 Troubleshooting

### "Training data not found"
- Make sure `data/train/` folder exists
- Check that `labels.csv` and `weather.csv` are present

### "Out of memory"
- Reduce `BATCH_SIZE` in `train.py` (try 8 or 4)
- Use smaller images (resize to 224×224 is already optimal)

### "Model not learning"
- Check that labels are correct (disease_label should be 0-9)
- Verify weather data has 14 rows per image
- Ensure images are clear and properly labeled

### "Predictions still random after training"
- Make sure you loaded the trained weights in `app.py`
- Check that `best_model.pth` exists
- Verify the model architecture matches (same num_diseases, etc.)

---

## 🌐 Next Steps

After training:

1. ✅ Load trained model in dashboard
2. ✅ Test with real crop images
3. ✅ Export to ONNX for edge deployment
4. ✅ Deploy to mobile devices or edge hardware
5. ✅ Collect feedback and retrain with more data

---

## 💡 Tips for Better Results

1. **Balanced Dataset**: Include equal samples of each disease type
2. **Data Augmentation**: Rotate, flip, and adjust brightness of images
3. **Quality Labels**: Double-check labels with agricultural experts
4. **Weather Accuracy**: Use precise weather station data
5. **Regular Retraining**: Update model as you collect more field data

---

## 📞 Need Help?

The model architecture is complete and tested. Training success depends on:
- Quality of your labeled data
- Sufficient training samples (500+ per disease class recommended)
- Accurate weather records
- Proper data organization

Good luck with training! 🚀
