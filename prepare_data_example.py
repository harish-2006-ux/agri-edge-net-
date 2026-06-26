"""
Example Data Preparation Script
================================
This script shows how to prepare your data for training.
Modify it to match your actual data source.
"""

import pandas as pd
from pathlib import Path
import shutil

def create_example_structure():
    """
    Create example data structure (for demonstration).
    Replace this with your actual data preparation logic.
    """
    
    print("Creating example data structure...")
    
    # Create directories
    for split in ['train', 'val']:
        (Path('data') / split / 'images').mkdir(parents=True, exist_ok=True)
    
    print("✓ Created directory structure:")
    print("  data/train/images/")
    print("  data/val/images/")
    
    # Create example labels.csv
    example_labels = {
        'train': pd.DataFrame({
            'image_name': ['img001.jpg', 'img002.jpg', 'img003.jpg'],
            'disease_label': [0, 2, 5],  # 0=Healthy, 2=Brown Spot, 5=Tungro
            'yield_value': [5.2, 4.5, 3.8],
            'cycle_label': [1, 1, 2],  # 1=Summer, 2=Fall
            'water': [0, 1, 1],
            'nitrogen': [0, 0, 1],
            'phosphorus': [0, 1, 0],
            'potassium': [0, 0, 1]
        }),
        'val': pd.DataFrame({
            'image_name': ['img004.jpg', 'img005.jpg'],
            'disease_label': [0, 3],
            'yield_value': [5.0, 4.2],
            'cycle_label': [1, 1],
            'water': [0, 1],
            'nitrogen': [0, 0],
            'phosphorus': [0, 1],
            'potassium': [0, 0]
        })
    }
    
    for split, df in example_labels.items():
        df.to_csv(f'data/{split}/labels.csv', index=False)
    
    print("\n✓ Created example labels.csv files")
    
    # Create example weather.csv
    for split in ['train', 'val']:
        weather_rows = []
        labels = example_labels[split]
        
        for img_name in labels['image_name']:
            for day in range(1, 15):  # 14 days
                weather_rows.append({
                    'image_name': img_name,
                    'day': day,
                    'temperature': 25.0 + day * 0.5,  # Example: increasing temp
                    'humidity': 60.0 + day,
                    'rainfall': 10.0 if day % 3 == 0 else 0.0
                })
        
        weather_df = pd.DataFrame(weather_rows)
        weather_df.to_csv(f'data/{split}/weather.csv', index=False)
    
    print("✓ Created example weather.csv files")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n1. Replace example data with your real agricultural images")
    print("   - Copy your images to data/train/images/ and data/val/images/")
    print("\n2. Update labels.csv with correct disease labels")
    print("   - disease_label: 0=Healthy, 1-9=different diseases")
    print("   - yield_value: Expected yield in tons/hectare")
    print("   - cycle_label: 0=Spring, 1=Summer, 2=Fall, 3=Winter")
    print("   - Resources: 1=needs adjustment, 0=no change needed")
    print("\n3. Update weather.csv with actual weather data")
    print("   - 14 rows per image (14-day history)")
    print("   - temperature (°C), humidity (%), rainfall (mm)")
    print("\n4. Run training:")
    print("   python core/train.py")
    print("\n" + "="*60)


def validate_data_structure():
    """
    Validate that data is properly structured.
    Run this before training.
    """
    
    print("Validating data structure...\n")
    
    errors = []
    
    for split in ['train', 'val']:
        # Check directories
        img_dir = Path('data') / split / 'images'
        if not img_dir.exists():
            errors.append(f"Missing: {img_dir}")
        
        # Check labels.csv
        labels_file = Path('data') / split / 'labels.csv'
        if not labels_file.exists():
            errors.append(f"Missing: {labels_file}")
        else:
            df = pd.read_csv(labels_file)
            required_cols = [
                'image_name', 'disease_label', 'yield_value', 'cycle_label',
                'water', 'nitrogen', 'phosphorus', 'potassium'
            ]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                errors.append(f"{labels_file}: Missing columns: {missing_cols}")
            
            # Check if images exist
            if img_dir.exists():
                for img_name in df['image_name']:
                    img_path = img_dir / img_name
                    if not img_path.exists():
                        errors.append(f"Image not found: {img_path}")
        
        # Check weather.csv
        weather_file = Path('data') / split / 'weather.csv'
        if not weather_file.exists():
            errors.append(f"Missing: {weather_file}")
        else:
            df = pd.read_csv(weather_file)
            required_cols = ['image_name', 'day', 'temperature', 'humidity', 'rainfall']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                errors.append(f"{weather_file}: Missing columns: {missing_cols}")
            
            # Check that each image has 14 weather rows
            weather_counts = df.groupby('image_name').size()
            invalid = weather_counts[weather_counts != 14]
            if len(invalid) > 0:
                errors.append(f"{weather_file}: Some images don't have exactly 14 weather rows")
    
    if errors:
        print("❌ VALIDATION FAILED\n")
        print("Errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ VALIDATION PASSED\n")
        print("Data structure is correct. You can run training now!")
        print("\nCommand:")
        print("  python core/train.py")
        return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        # Validate existing data
        validate_data_structure()
    else:
        # Create example structure
        create_example_structure()
        print("\nTo validate your data structure, run:")
        print("  python prepare_data_example.py validate")
