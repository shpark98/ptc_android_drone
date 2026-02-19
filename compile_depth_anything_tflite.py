#!/usr/bin/env python3
"""
Compile Depth Anything V2 for TFLite with QNN delegate (DSP/HTP acceleration)
Based on official Qualcomm AI Hub documentation
"""

import sys
import qai_hub
import qai_hub_models
from qai_hub_models.models.depth_anything_v2 import Model

print("=" * 60)
print("Qualcomm AI Hub - Depth Anything V2 TFLite Compilation")
print("Target: TFLite with QNN Delegate (Hexagon DSP)")
print("=" * 60)

# Get Samsung Galaxy S22
devices = qai_hub.get_devices()
target_device = None
for device in devices:
    if device.name == "Samsung Galaxy S22 (Family)":
        target_device = device
        print(f"\n✓ Found target: {device.name}")
        print(f"  Chipset: SM8450 (Snapdragon 8 Gen 1)")
        print(f"  Hexagon: v69")
        break

if not target_device:
    print("❌ Device not found")
    sys.exit(1)

# Load model
print("\n🔧 Loading Depth Anything V2 model...")
model = Model.from_pretrained()
print("  ✓ Model loaded")

# Convert to TorchScript (required for compilation)
print("\n📦 Converting to TorchScript...")
traced_model = model.convert_to_torchscript(model.get_input_spec())
print("  ✓ Model traced")

# Compile for TFLite with QNN backend
print(f"\n⚙️ Submitting TFLite compilation job...")
print("  Target runtime: TFLite")
print("  Backend: QNN (Hexagon HTP/DSP)")
print("  This will take 5-10 minutes...")

try:
    compile_job = qai_hub.submit_compile_job(
        model=traced_model,
        device=target_device,
        input_specs=model.get_input_spec(),
        options="--target_runtime tflite"  # TFLite format with QNN optimization
    )

    print(f"\n  ✓ Job submitted: {compile_job.job_id}")
    print(f"  URL: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
    print("\n  Waiting for compilation...")

    compile_job.wait()
    status = compile_job.get_status()
    print(f"\n✓ Compilation completed!")
    print(f"  Status: {status}")

    # Download compiled model
    output_path = "weights/depth_anything_v2_onnx/depth_anything_v2_qnn.tflite"
    print(f"\n📥 Downloading TFLite model to {output_path}...")
    compile_job.download_target_model(output_path)

    print(f"\n✅ TFLite model saved: {output_path}")
    print("\nNext steps:")
    print(f"  1. Copy to Android assets:")
    print(f"     cp {output_path} android/app/src/main/assets/depth_anything_v2_qnn.tflite")
    print("  2. Update MainActivity.kt:")
    print("     - 우선순위를 TFLite로 변경")
    print("  3. Rebuild and test")
    print("\n🔥 Expected: ~50-100ms inference with QNN delegate on Hexagon DSP")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
