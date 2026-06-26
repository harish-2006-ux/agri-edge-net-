"""
PlantVillage Dataset Downloader and Preparer
=============================================
Downloads and prepares the PlantVillage dataset for EdgeAgri-Net training.

Dataset: 54,000+ images, 38 disease classes, 14 plant species
- Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato,
  Raspberry, Soybean, Squash, Strawberry, Tomato

Source: https://www.kaggle.com/datasets/emmarex/plantdisease
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import random
from PIL import Image


def download_instructions():
    """Provide instructions to download PlantVillage dataset."""
    print("\n" + "="*70)
    print("PlantVillage Dataset Download Instructions")
    print("="*70)
    
    print("\n📥 STEP 1: Download the Dataset")
    print("-" * 70)
    print("\n1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("2. Click 'Download' button (requires free Kaggle account)")
    print("3. Save the ZIP file (plantdisease.zip, ~2.5 GB)")
    print("4. Extract it to a folder")
    
    print("\n📂 Expected structure after extraction:")
    print("   PlantVillage/")
    print("   ├── Apple___Apple_scab/")
    print("   ├── Apple___Black_rot/")
    print("   ├── Apple___Cedar_apple_rust/")
    print("   ├── Apple___healthy/")
    print("   ├── Tomato___Bacterial_spot/")
    print("   └── ... (38 total classes)")
    
    print("\n" + "="*70)
    print("After downloading and extracting, run this script again!")
    print("="*70)


def prepare_plantvillage_data(source_dir: str, use_all_plants: bool = True):
    """
    Prepare PlantVillage data for EdgeAgri-Net training.
    
    Args:
        source_dir: Path to extracted PlantVillage folder
        use_all_plants: If True, use all 14 plants. If False, select specific plants.
    """
    
    if not os.path.exists(source_dir):
        print(f"❌ Error: Directory '{source_dir}' not found!")
        print("\nDid you extract the dataset? It should contain folders like:")
        print("  Apple___Apple_scab/, Tomato___Bacterial_spot/, etc.")
        return
    
    # Get all class folders
    all_folders = [f for f in os.listdir(source_dir) 
                   if os.path.isdir(os.path.join(source_dir, f)) and '___' in f]
    
    if not all_folders:
        print("❌ Error: No plant disease folders found!")
        print("Expected folder names like: Apple___Apple_scab")
        return
    
    print(f"\n✓ Found {len(all_folders)} disease classes")
    
    # Organize by plant species
    plants_dict = {}
    for folder in all_folders:
        plant, disease = folder.split('___')
        if plant not in plants_dict:
            plants_dict[plant] = []
        plants_dict[plant].append({'folder': folder, 'disease': disease})
    
    print(f"\n✓ Found {len(plants_dict)} plant species:")
    for plant, diseases in sorted(plants_dict.items()):
        print(f"  {plant}: {len(diseases)} disease classes")
    
    # Let user select plants
    selected_folders = []
    
    if use_all_plants:
        print("\n📦 Using ALL plant species")
        selected_folders = all_folders
    else:
        print("\n🌱 Select which plants to include:")
        for i, (plant, diseases) in enumerate(sorted(plants_dict.items()), 1):
            print(f"  {i}. {plant} ({len(diseases)} diseases)")
        
        selection = input("\nEnter plant numbers (comma-separated, or 'all'): ").strip()
        
        if selection.lower() == 'all':
            selected_folders = all_folders
        else:
            selected_plants = []
            for num in selection.split(','):
                try:
                    idx = int(num.strip()) - 1
                    plant_name = sorted(plants_dict.keys())[idx]
                    selected_plants.append(plant_name)
                except:
                    pass
            
            for plant in selected_plants:
                for disease_info in plants_dict[plant]:
                    selected_folders.append(disease_info['folder'])
    
    print(f"\n✓ Selected {len(selected_folders)} disease classes")
    
    # Create disease label mapping
    disease_map = {folder: idx for idx, folder in enumerate(sorted(selected_folders))}
    
    print("\n" + "="*70)
    print("Disease Label Mapping")
    print("="*70)
    for folder, label in sorted(disease_map.items(), key=lambda x: x[1]):
        plant, disease = folder.split('___')
        img_count = len([f for f in os.listdir(os.path.join(source_dir, folder))
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {label:3d}: {plant:15s} - {disease:30s} ({img_count} images)")
    
    # Save mapping to file
    with open('plant_disease_mapping.txt', 'w') as f:
        f.write("Disease Label Mapping\n")
        f.write("=" * 70 + "\n")
        for folder, label in sorted(disease_map.items(), key=lambda x: x[1]):
            plant, disease = folder.split('___')
            f.write(f"{label:3d}: {plant:15s} - {disease}\n")
    
    print("\n✓ Saved mapping to: plant_disease_mapping.txt")
    
    # Prepare training data
    print("\n" + "="*70)
    print("Preparing Training Data")
    print("="*70)
    
    Path('data/train/images').mkdir(parents=True, exist_ok=True)
    Path('data/val/images').mkdir(parents=True, exist_ok=True)
    
    # Collect all images
    all_data = []
    
    for folder, disease_label in disease_map.items():
        folder_path = os.path.join(source_dir, folder)
        images = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
        
        plant, disease = folder.split('___')
        
        for img_file in images:
            src_path = os.path.join(folder_path, img_file)
            all_data.append({
                'src_path': src_path,
                'disease_label': disease_label,
                'plant': plant,
                'disease': disease,
                'folder': folder
            })
    
    print(f"\n✓ Collected {len(all_data)} total images")
    
    # Shuffle and split 80/20
    random.seed(42)  # For reproducibility
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"✓ Split: {len(train_data)} train, {len(val_data)} validation")
    
    # Process and copy images
    print("\n📋 Processing images (this may take a few minutes)...")
    
    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        labels_rows = []
        weather_rows = []
        
        print(f"\nProcessing {split_name} set ({len(split_data)} images)...")
        
        for idx, item in enumerate(split_data):
            if idx % 1000 == 0:
                print(f"  Progress: {idx}/{len(split_data)}")
            
            # New filename
            new_filename = f"{item['plant'].lower()}_{split_name}_{idx:05d}.jpg"
            dst_path = f"data/{split_name}/images/{new_filename}"
            
            # Copy and resize image to 224x224 (saves space and training time)
            try:
                img = Image.open(item['src_path']).convert('RGB')
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img.save(dst_path, 'JPEG', quality=95)
            except Exception as e:
                print(f"⚠️ Warning: Could not process {item['src_path']}: {e}")
                continue
            
            # Determine if healthy (for better default values)
            is_healthy = 'healthy' in item['disease'].lower()
            
            # Create label row with estimated values
            labels_rows.append({
                'image_name': new_filename,
                'disease_label': item['disease_label'],
                'yield_value': 5.0 if is_healthy else random.uniform(2.5, 4.5),
                'cycle_label': random.randint(0, 3),
                'water': 0 if is_healthy else random.randint(0, 1),
                'nitrogen': 0 if is_healthy else random.randint(0, 1),
                'phosphorus': 0 if is_healthy else random.randint(0, 1),
                'potassium': 0 if is_healthy else random.randint(0, 1)
            })
            
            # Create 14-day weather data (simulated)
            base_temp = random.uniform(20, 30)
            base_humidity = random.uniform(55, 75)
            
            for day in range(1, 15):
                weather_rows.append({
                    'image_name': new_filename,
                    'day': day,
                    'temperature': base_temp + random.uniform(-3, 3),
                    'humidity': base_humidity + random.uniform(-10, 10),
                    'rainfall': random.uniform(0, 25) if random.random() > 0.6 else 0
                })
        
        # Save CSVs
        pd.DataFrame(labels_rows).to_csv(f'data/{split_name}/labels.csv', index=False)
        pd.DataFrame(weather_rows).to_csv(f'data/{split_name}/weather.csv', index=False)
        
        print(f"✓ Saved {len(labels_rows)} images to data/{split_name}/")
    
    print("\n" + "="*70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nDataset Statistics:")
    print(f"  Plant species: {len(plants_dict)}")
    print(f"  Disease classes: {len(disease_map)}")
    print(f"  Total images: {len(all_data)}")
    print(f"    - Training: {len(train_data)}")
    print(f"    - Validation: {len(val_data)}")
    
    print("\n⚠️ IMPORTANT NOTES:")
    print("  - Weather data is SIMULATED (random realistic values)")
    print("  - Yield/prescriptions are ESTIMATED based on disease presence")
    print("  - Images resized to 224x224 for faster training")
    
    print("\n📝 BEFORE TRAINING:")
    print("  1. Open core/train.py")
    print(f"  2. Change 'num_diseases=10' to 'num_diseases={len(disease_map)}'")
    print("  3. Update dashboard disease labels (see update_dashboard.py)")
    
    print("\n🚀 START TRAINING:")
    print("  python core/train.py")
    
    # Create quick reference file
    with open('QUICK_START.txt', 'w') as f:
        f.write(f"EdgeAgri-Net Quick Start\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Dataset: PlantVillage\n")
        f.write(f"Disease classes: {len(disease_map)}\n")
        f.write(f"Training images: {len(train_data)}\n")
        f.write(f"Validation images: {len(val_data)}\n\n")
        f.write(f"STEP 1: Update core/train.py\n")
        f.write(f"  Line ~216: num_diseases={len(disease_map)}\n\n")
        f.write(f"STEP 2: Start training\n")
        f.write(f"  python core/train.py\n\n")
        f.write(f"STEP 3: After training, update dashboard\n")
        f.write(f"  python update_dashboard.py\n")
    
    print("\n✓ Created QUICK_START.txt with instructions")


def main():
    """Main menu."""
    print("\n" + "="*70)
    print("PlantVillage Dataset Preparation for EdgeAgri-Net")
    print("="*70)
    print("\nThis will prepare 54,000+ images across 14 plant species:")
    print("  Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach,")
    print("  Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato")
    
    choice = input("\n1. Show download instructions\n2. Prepare dataset (already downloaded)\n\nChoice (1/2): ").strip()
    
    if choice == '1':
        download_instructions()
    elif choice == '2':
        source_dir = input("\nEnter path to extracted PlantVillage folder: ").strip()
        use_all = input("Use all 14 plants? (y/n): ").strip().lower() == 'y'
        prepare_plantvillage_data(source_dir, use_all)
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
