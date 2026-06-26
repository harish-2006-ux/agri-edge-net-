"""
EdgeAgri-Net Training Script
=============================
Train the model on your labeled agricultural dataset.

Dataset Structure Expected:
---------------------------
data/
├── train/
│   ├── images/          # Leaf images (*.jpg, *.png)
│   ├── weather.csv      # Weather data (14 days per image)
│   └── labels.csv       # Labels: disease, yield, cycle, prescriptions
└── val/
    ├── images/
    ├── weather.csv
    └── labels.csv

labels.csv format:
-----------------
image_name,disease_label,yield_value,cycle_label,water,nitrogen,phosphorus,potassium
img001.jpg,2,4.5,1,1,0,1,0
img002.jpg,0,5.2,2,0,0,0,0
...
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

from edgeagrinet_core import EdgeAgriNet, MultiTaskLoss


class AgriDataset(Dataset):
    """Custom dataset for agricultural data."""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load labels
        self.labels_df = pd.read_csv(self.data_dir / "labels.csv")
        
        # Load weather data
        self.weather_df = pd.read_csv(self.data_dir / "weather.csv")
        
        self.image_dir = self.data_dir / "images"
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row['image_name']
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))
        
        # Convert to tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Load weather data for this image
        weather_rows = self.weather_df[
            self.weather_df['image_name'] == row['image_name']
        ]
        weather_data = weather_rows[['temperature', 'humidity', 'rainfall']].values
        weather_tensor = torch.tensor(weather_data, dtype=torch.float32)
        
        # Labels
        targets = {
            'disease': torch.tensor(row['disease_label'], dtype=torch.long),
            'yield': torch.tensor(row['yield_value'], dtype=torch.float32),
            'cycle': torch.tensor(row['cycle_label'], dtype=torch.long),
            'prescription': torch.tensor([
                row['water'],
                row['nitrogen'],
                row['phosphorus'],
                row['potassium']
            ], dtype=torch.long)
        }
        
        return img_tensor, weather_tensor, targets


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    task_losses = {'disease': 0, 'yield': 0, 'cycle': 0, 'prescription': 0}
    
    pbar = tqdm(dataloader, desc="Training")
    for images, weather, targets in pbar:
        # Move to device
        images = images.to(device)
        weather = weather.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, weather)
        
        # Compute loss
        loss, losses_dict = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for k, v in losses_dict.items():
            task_losses[k.replace('loss_', '')] += v
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_task_losses = {k: v / len(dataloader) for k, v in task_losses.items()}
    
    return avg_loss, avg_task_losses


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    task_losses = {'disease': 0, 'yield': 0, 'cycle': 0, 'prescription': 0}
    
    with torch.no_grad():
        for images, weather, targets in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            weather = weather.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            outputs = model(images, weather)
            loss, losses_dict = criterion(outputs, targets)
            
            total_loss += loss.item()
            for k, v in losses_dict.items():
                task_losses[k.replace('loss_', '')] += v
    
    avg_loss = total_loss / len(dataloader)
    avg_task_losses = {k: v / len(dataloader) for k, v in task_losses.items()}
    
    return avg_loss, avg_task_losses


def main():
    """Main training loop."""
    
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("EdgeAgri-Net Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Check for data directory
    if not Path("data/train").exists():
        print("\n⚠️ ERROR: Training data not found!")
        print("\nExpected structure:")
        print("  data/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   ├── weather.csv")
        print("  │   └── labels.csv")
        print("  └── val/")
        print("      ├── images/")
        print("      ├── weather.csv")
        print("      └── labels.csv")
        print("\nPlease prepare your dataset and try again.")
        return
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = AgriDataset("data/train")
    val_dataset = AgriDataset("data/val")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = EdgeAgriNet(
        num_diseases=10,  # Adjust based on your dataset
        num_cycles=4,
        num_resources=4
    )
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = MultiTaskLoss(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_task_losses = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_task_losses = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"  Disease: {train_task_losses['disease']:.4f}")
        print(f"  Yield: {train_task_losses['yield']:.4f}")
        print(f"  Cycle: {train_task_losses['cycle']:.4f}")
        print(f"  Prescription: {train_task_losses['prescription']:.4f}")
        
        print(f"\nVal Loss: {val_loss:.4f}")
        print(f"  Disease: {val_task_losses['disease']:.4f}")
        print(f"  Yield: {val_task_losses['yield']:.4f}")
        print(f"  Cycle: {val_task_losses['cycle']:.4f}")
        print(f"  Prescription: {val_task_losses['prescription']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            print("\n✓ Saved best model!")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Model saved to: best_model.pth")


if __name__ == "__main__":
    main()
