"""
EdgeAgri-Net ONNX Compilation Pipeline
=======================================
Export the PyTorch model to ONNX format for edge deployment.

Features:
- Trace and export model to ONNX
- INT8 post-training quantization (optional)
- Validation of exported model
"""

import torch
import torch.onnx
from pathlib import Path
import numpy as np

from edgeagrinet_core import EdgeAgriNet


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str = "edgeagrinet.onnx",
    opset_version: int = 14,
    dynamic_batch: bool = True
):
    """
    Export EdgeAgri-Net model to ONNX format.
    
    Args:
        model: Trained EdgeAgriNet model
        output_path: Path to save ONNX file
        opset_version: ONNX opset version
        dynamic_batch: Whether to support dynamic batch size
    """
    
    model.eval()
    
    # Create dummy inputs
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_weather = torch.randn(1, 14, 3)
    
    # Dynamic axes for variable batch size
    if dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'weather': {0: 'batch_size'},
            'disease_logits': {0: 'batch_size'},
            'yield_pred': {0: 'batch_size'},
            'cycle_logits': {0: 'batch_size'},
            'prescription_logits': {0: 'batch_size'}
        }
    else:
        dynamic_axes = None
    
    # Export
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_image, dummy_weather),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image', 'weather'],
        output_names=['disease_logits', 'yield_pred', 'cycle_logits', 'prescription_logits'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"✓ Model exported successfully to {output_path}")
    
    # Get file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    return output_path


def validate_onnx_model(onnx_path: str):
    """
    Validate exported ONNX model.
    
    Args:
        onnx_path: Path to ONNX file
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check ONNX model
        print(f"\nValidating ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure is valid")
        
        # Test inference with ONNX Runtime
        print("\nTesting ONNX Runtime inference...")
        ort_session = ort.InferenceSession(onnx_path)
        
        # Create test inputs
        test_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
        test_weather = np.random.randn(1, 14, 3).astype(np.float32)
        
        # Run inference
        outputs = ort_session.run(
            None,
            {
                'image': test_image,
                'weather': test_weather
            }
        )
        
        print("✓ ONNX Runtime inference successful")
        print(f"  Output shapes:")
        print(f"    disease_logits: {outputs[0].shape}")
        print(f"    yield_pred: {outputs[1].shape}")
        print(f"    cycle_logits: {outputs[2].shape}")
        print(f"    prescription_logits: {outputs[3].shape}")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Validation skipped: {e}")
        print("  Install onnx and onnxruntime for validation:")
        print("  pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def quantize_onnx_model(onnx_path: str, output_path: str = None):
    """
    Apply INT8 post-training quantization to ONNX model.
    
    Args:
        onnx_path: Path to input ONNX file
        output_path: Path to save quantized model (default: adds '_int8' suffix)
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if output_path is None:
            output_path = onnx_path.replace('.onnx', '_int8.onnx')
        
        print(f"\nQuantizing model to INT8...")
        quantize_dynamic(
            model_input=onnx_path,
            model_output=output_path,
            weight_type=QuantType.QInt8
        )
        
        # Compare file sizes
        original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"✓ Quantized model saved to {output_path}")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")
        
        return output_path
        
    except ImportError:
        print("⚠ Quantization skipped: onnxruntime not installed")
        print("  Install with: pip install onnxruntime")
        return None
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        return None


def main():
    """Main compilation pipeline."""
    print("=" * 60)
    print("EdgeAgri-Net ONNX Compilation Pipeline")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(__file__).parent / "exports"
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("\n1. Loading EdgeAgri-Net model...")
    model = EdgeAgriNet(
        num_diseases=10,
        num_cycles=4,
        num_resources=4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded ({total_params:,} parameters)")
    
    # Export to ONNX
    print("\n2. Exporting to ONNX format...")
    onnx_path = str(output_dir / "edgeagrinet.onnx")
    export_to_onnx(model, onnx_path)
    
    # Validate
    print("\n3. Validating ONNX model...")
    validate_onnx_model(onnx_path)
    
    # Optional: Quantize
    print("\n4. Applying INT8 quantization...")
    quantized_path = quantize_onnx_model(onnx_path)
    
    if quantized_path:
        print("\n5. Validating quantized model...")
        validate_onnx_model(quantized_path)
    
    print("\n" + "=" * 60)
    print("✓ Compilation pipeline complete!")
    print("=" * 60)
    print(f"\nExported files:")
    print(f"  FP32: {onnx_path}")
    if quantized_path:
        print(f"  INT8: {quantized_path}")
    print(f"\nTo use the exported model:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{onnx_path}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
