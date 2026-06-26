"""
EdgeAgri-Net: Lightweight Multimodal Cross-Attention Network
=============================================================
A multi-task deep learning platform for precision agriculture.

Inputs:
    - Leaf camera images (spatial)
    - 14-day weather metrics (temporal: temperature, humidity, rain)

Outputs:
    - Disease diagnosis (classification)
    - Yield volume forecast (regression)
    - Planting cycle selection (classification)
    - Resource prescription (multi-label classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


class TemporalEncoder(nn.Module):
    """1D Causal Temporal Convolutional Network for weather data."""
    
    def __init__(self, input_dim: int = 3, embed_dim: int = 512):
        """
        Args:
            input_dim: Number of weather features (temp, humidity, rain = 3)
            embed_dim: Output embedding dimension
        """
        super().__init__()
        
        # Causal convolution ensures no future data leaks into predictions
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, embed_dim, kernel_size=3, padding=1)
        
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(256)
        self.norm3 = nn.BatchNorm1d(embed_dim)
        
        # Use global average pooling (ONNX compatible)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Weather data of shape (batch, 14, input_dim)
        Returns:
            Temporal embeddings of shape (batch, embed_dim)
        """
        # (batch, 14, input_dim) -> (batch, input_dim, 14)
        x = x.transpose(1, 2)
        
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        x = F.silu(self.norm3(self.conv3(x)))
        
        # (batch, embed_dim, 1) -> (batch, embed_dim)
        x = self.pool(x).squeeze(-1)
        
        return x


class VisionEncoder(nn.Module):
    """MobileNetV3-Small backbone for image feature extraction."""
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        
        # Load MobileNetV3-Small (edge-optimized, no pretrained weights)
        # Note: Set weights=MobileNet_V3_Small_Weights.DEFAULT when network is available
        mobilenet = models.mobilenet_v3_small(weights=None)
        
        # Extract feature extractor (removes classification head)
        # MobileNetV3-Small outputs 576 channels before the classifier
        self.features = mobilenet.features
        
        # Project to desired embedding dimension
        self.project = nn.Conv2d(576, embed_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images of shape (batch, 3, H, W)
        Returns:
            Visual feature maps of shape (batch, embed_dim, H', W')
        """
        x = self.features(x)
        x = self.project(x)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Core
    ============================
    Weather embeddings (Q) attend to visual feature maps (K, V).
    
    Defense 1: Dual LayerNorm on both inputs before attention
    Defense 2: AdaptiveAvgPool2d((1,1)) on visual features before attention
    """
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        
        # DEFENSE 1: Separate LayerNorm for each modality
        self.image_norm = nn.LayerNorm(embed_dim)
        self.weather_norm = nn.LayerNorm(embed_dim)
        
        # DEFENSE 2: AdaptiveAvgPool2d to prevent attention smearing
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, image_features: torch.Tensor, weather_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: Visual features (batch, embed_dim, H, W)
            weather_embeds: Temporal embeddings (batch, embed_dim)
        Returns:
            Fused context vector (batch, embed_dim)
        """
        batch_size = image_features.size(0)
        
        # DEFENSE 2: Pool spatial dimensions to prevent smearing
        # (batch, embed_dim, H, W) -> (batch, embed_dim, 1, 1) -> (batch, embed_dim)
        pooled_image = self.adaptive_pool(image_features).view(batch_size, -1)
        
        # DEFENSE 1: Normalize both modalities
        image_normed = self.image_norm(pooled_image)
        weather_normed = self.weather_norm(weather_embeds)
        
        # Cross-attention: Weather (Q) attends to Image (K, V)
        # Query from weather, Key/Value from image
        q = weather_normed.unsqueeze(1)  # (batch, 1, embed_dim)
        k = image_normed.unsqueeze(1)
        v = image_normed.unsqueeze(1)
        
        attn_output, _ = self.cross_attention(q, k, v)
        attn_output = attn_output.squeeze(1)  # (batch, embed_dim)
        
        # Residual connection + FFN
        fused = self.ffn(self.ffn_norm(attn_output + weather_normed))
        
        return fused


class TaskHeads(nn.Module):
    """Multi-task output heads for 4 parallel predictions."""
    
    def __init__(self, embed_dim: int = 512, num_diseases: int = 10, 
                 num_cycles: int = 4, num_resources: int = 4):
        super().__init__()
        
        # Disease classification (Cross-Entropy)
        self.disease_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_diseases)
        )
        
        # Yield regression (Huber loss applied during training)
        self.yield_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Planting cycle selection (Cross-Entropy)
        self.cycle_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_cycles)
        )
        
        # Resource prescription (multi-label Binary Cross-Entropy)
        self.prescription_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_resources)
        )
        
    def forward(self, fused_context: torch.Tensor):
        """
        Args:
            fused_context: (batch, embed_dim)
        Returns:
            dict with keys: disease_logits, yield_pred, cycle_logits, prescription_logits
        """
        return {
            'disease_logits': self.disease_head(fused_context),
            'yield_pred': self.yield_head(fused_context),
            'cycle_logits': self.cycle_head(fused_context),
            'prescription_logits': self.prescription_head(fused_context)
        }


class EdgeAgriNet(nn.Module):
    """
    EdgeAgri-Net: Lightweight Multimodal Cross-Attention Network
    ==============================================================
    
    A compilable multi-task PyTorch model for precision agriculture.
    """
    
    def __init__(self, 
                 num_diseases: int = 10,
                 num_cycles: int = 4,
                 num_resources: int = 4,
                 weather_features: int = 3,
                 embed_dim: int = 512):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Encoders
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.temporal_encoder = TemporalEncoder(
            input_dim=weather_features, 
            embed_dim=embed_dim
        )
        
        # Fusion
        self.cross_attention = CrossAttentionFusion(embed_dim=embed_dim)
        
        # Task heads
        self.task_heads = TaskHeads(
            embed_dim=embed_dim,
            num_diseases=num_diseases,
            num_cycles=num_cycles,
            num_resources=num_resources
        )
        
    def forward(self, image: torch.Tensor, weather: torch.Tensor):
        """
        Full forward pass.
        
        Args:
            image: Leaf images (batch, 3, 224, 224)
            weather: 14-day weather metrics (batch, 14, 3) -> (batch, days, features)
        Returns:
            dict with 4 task outputs
        """
        # Encode both modalities
        image_features = self.vision_encoder(image)
        weather_embeds = self.temporal_encoder(weather)
        
        # Fuse via cross-attention
        fused_context = self.cross_attention(image_features, weather_embeds)
        
        # Generate multi-task outputs
        outputs = self.task_heads(fused_context)
        
        return outputs
    
    def predict(self, image: torch.Tensor, weather: torch.Tensor):
        """
        Inference mode: returns human-readable predictions.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image, weather)
            
            return {
                'disease': outputs['disease_logits'].argmax(dim=-1),
                'disease_probs': F.softmax(outputs['disease_logits'], dim=-1),
                'yield_tons_per_hectare': outputs['yield_pred'].squeeze(-1),
                'cycle': outputs['cycle_logits'].argmax(dim=-1),
                'cycle_probs': F.softmax(outputs['cycle_logits'], dim=-1),
                'prescription': torch.sigmoid(outputs['prescription_logits']) > 0.5
            }


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Joint Optimization Loss
    ===================================
    Combined loss with task-specific loss functions.
    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        super().__init__()
        
        # Loss weights (can be learned or tuned)
        self.alpha = alpha  # disease
        self.beta = beta    # yield
        self.gamma = gamma  # cycles
        self.delta = delta  # prescription
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from model
            targets: dict with keys disease, yield, cycle, prescription
        Returns:
            Total weighted loss
        """
        # Disease: Cross-Entropy
        loss_disease = F.cross_entropy(
            outputs['disease_logits'], 
            targets['disease']
        )
        
        # Yield: Huber Loss (smooth L1, robust to outliers)
        loss_yield = F.huber_loss(
            outputs['yield_pred'].squeeze(-1),
            targets['yield']
        )
        
        # Cycle: Cross-Entropy
        loss_cycle = F.cross_entropy(
            outputs['cycle_logits'],
            targets['cycle']
        )
        
        # Prescription: Binary Cross-Entropy with Logits
        loss_prescription = F.binary_cross_entropy_with_logits(
            outputs['prescription_logits'],
            targets['prescription'].float()
        )
        
        # Weighted sum
        total = (self.alpha * loss_disease + 
                 self.beta * loss_yield + 
                 self.gamma * loss_cycle + 
                 self.delta * loss_prescription)
        
        return total, {
            'loss_disease': loss_disease.item(),
            'loss_yield': loss_yield.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_prescription': loss_prescription.item()
        }


# ---------------------------------------------------------------------------
# Market Simulation (Economic Planning Module)
# ---------------------------------------------------------------------------

def market_simulation(yield_hat: float, price_current: float, 
                      delta_p: float, production_cost: float) -> float:
    """
    Dynamic Market-Price Elasticity & Simulation Engine.
    
    R_sim(Delta P) = [Y_hat * (P_current * (1 + Delta_P/100))] - C_production
    
    Args:
        yield_hat: Predicted yield (tons/hectare)
        price_current: Live wholesale price ($/ton)
        delta_p: Percentage shift (-30 to +30)
        production_cost: Fixed production cost ($)
    
    Returns:
        Simulated revenue ($)
    """
    adjusted_price = price_current * (1 + delta_p / 100)
    revenue = (yield_hat * adjusted_price) - production_cost
    return revenue


# ---------------------------------------------------------------------------
# Example Usage & Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create model
    model = EdgeAgriNet(
        num_diseases=10,
        num_cycles=4,
        num_resources=4
    )
    
    # Dummy inputs (batch_size=2)
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_weather = torch.randn(2, 14, 3)  # 14 days, 3 features
    
    # Forward pass
    outputs = model(dummy_image, dummy_weather)
    
    print("EdgeAgri-Net Output Shapes:")
    print(f"  disease_logits: {outputs['disease_logits'].shape}")
    print(f"  yield_pred:     {outputs['yield_pred'].shape}")
    print(f"  cycle_logits:   {outputs['cycle_logits'].shape}")
    print(f"  prescription:   {outputs['prescription_logits'].shape}")
    
    # Test market simulation
    revenue = market_simulation(
        yield_hat=4.2,
        price_current=350,
        delta_p=10.0,
        production_cost=800
    )
    print(f"\nMarket Simulation (10% price increase):")
    print(f"  Revenue: ${revenue:.2f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")