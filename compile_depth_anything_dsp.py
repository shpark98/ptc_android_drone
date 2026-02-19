#!/usr/bin/env python3
"""
Compile Depth Anything V2 for Qualcomm DSP/HTP runtime
Targets: Snapdragon 8 Gen 1 (SM8450)
"""

import qai_hub
import qai_hub_models
from qai_hub_models.models.depth_anything_v2 import Model

print("=" * 60)
print("Qualcomm AI Hub - Depth Anything V2 DSP Compilation")
print("=" * 60)

# Check login
try:
    devices = qai_hub.get_devices()
    print(f"✓ Logged in to Qualcomm AI Hub")
except Exception as e:
    print(f"✗ Not logged in: {e}")
    print("\nPlease run: qai-hub configure")
    print("And follow the prompts to log in with your Qualcomm ID")
    exit(1)

# Find Snapdragon 8 Gen 1 device
print("\n📱 Looking for Snapdragon 8 Gen 1 device...")
target_device = None
for device in devices:
    print(f"  - {device.name} (ID: {device.id})")
    if "8gen1" in device.name.lower() or "sm8450" in device.name.lower():
        target_device = device
        print(f"  ✓ Found target: {device.name}")

if not target_device:
    print("\n⚠️ Snapdragon 8 Gen 1 not found. Using first available device.")
    target_device = devices[0]
    print(f"  Using: {target_device.name}")

# Load Depth Anything V2 model
print("\n🔧 Loading Depth Anything V2 model...")
model = Model.from_pretrained()

# Compile for DSP/HTP
print(f"\n⚙️ Compiling for DSP/HTP on {target_device.name}...")
print("  This may take several minutes...")

compile_job = qai_hub.submit_compile_job(
    model=model.get_model(),
    device=target_device,
    input_specs=model.get_input_spec(),
    options="--target_runtime htp"  # Force HTP (Hexagon Tensor Processor / DSP)
)

print(f"  Compile job submitted: {compile_job.job_id}")
print("  Waiting for compilation to complete...")

# Wait for job to complete
compile_job.wait()

if compile_job.is_failed():
    print(f"\n❌ Compilation failed:")
    print(compile_job.get_status())
    exit(1)

# Download compiled DLC
print(f"\n✓ Compilation succeeded!")
output_path = "weights/depth_anything_v2_onnx/depth_anything_v2_dsp.dlc"
compile_job.download_target_model(output_path)

print(f"\n✅ DSP-optimized DLC saved to: {output_path}")
print("\nNext steps:")
print(f"  1. Copy {output_path} to android/app/src/main/assets/")
print("  2. Update DepthEstimatorSNPE.kt to use 'depth_anything_v2_dsp.dlc'")
print("  3. Rebuild and test the app")
print("\n🔥 Expected performance: ~50ms inference on Hexagon 780 DSP")
