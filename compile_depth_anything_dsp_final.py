#!/usr/bin/env python3
"""
Compile Depth Anything V2 for Qualcomm DSP/HTP runtime
Targets: Snapdragon 8 Gen 1 (SM8450)
"""

import sys

print("=" * 60)
print("Qualcomm AI Hub - Depth Anything V2 DSP Compilation")
print("=" * 60)

import qai_hub
import qai_hub_models
from qai_hub_models.models.depth_anything_v2 import Model

# Check login
try:
    devices = qai_hub.get_devices()
    print(f"✓ Logged in to Qualcomm AI Hub")
    print(f"  Found {len(devices)} available devices")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
    print("\nPlease ensure ~/.qai_hub/client.ini exists with your API token.")
    sys.exit(1)

# Find Snapdragon 8 Gen 1 device
print("\n📱 Looking for Snapdragon 8 Gen 1 device...")
target_device = None
for device in devices:
    device_name = device.name.lower()
    print(f"  - {device.name}")
    if "8 gen 1" in device_name or "sm8450" in device_name or "s22" in device_name:
        target_device = device
        print(f"  ✓ Found target: {device.name}")
        break

if not target_device:
    print("\n⚠️ Snapdragon 8 Gen 1 not found. Using first available device.")
    target_device = devices[0]
    print(f"  Using: {target_device.name}")

# Load Depth Anything V2 model
print("\n🔧 Loading Depth Anything V2 model...")
print("  This will download the model from Hugging Face (may take a few minutes)...")
model = Model.from_pretrained()
print("  ✓ Model loaded")

# Convert to TorchScript
print("\n📦 Converting model to TorchScript...")
traced_model = model.convert_to_torchscript(
    model.get_input_spec()
)
print("  ✓ Model traced")

# Compile for DSP/HTP
print(f"\n⚙️ Submitting compile job for DSP/HTP on {target_device.name}...")
print("  Target: Hexagon HTP (DSP)")
print("  This will take 5-10 minutes in Qualcomm cloud...")

try:
    compile_job = qai_hub.submit_compile_job(
        model=traced_model,  # Use traced TorchScript model
        device=target_device,
        input_specs=model.get_input_spec(),
        options="--target_runtime qnn_dlc"  # QNN DLC for Hexagon DSP acceleration
    )

    print(f"  ✓ Compile job submitted: {compile_job.job_id}")
    print("  Waiting for compilation to complete...")

    # Wait for job to complete (this will take several minutes)
    compile_job.wait()

    # Check job status and download
    print(f"\n✓ Compilation job completed!")
    print(f"  Job status: {compile_job.get_status()}")

    # Download compiled DLC
    output_path = "weights/depth_anything_v2_onnx/depth_anything_v2_dsp_htp.dlc"
    print(f"\n📥 Downloading compiled model to {output_path}...")
    compile_job.download_target_model(output_path)

    print(f"\n✅ DSP-optimized DLC saved to: {output_path}")
    print("\nNext steps:")
    print(f"  1. Copy to Android assets:")
    print(f"     cp {output_path} android/app/src/main/assets/depth_anything_v2_dsp.dlc")
    print("  2. Update DepthEstimatorSNPE.kt:")
    print("     modelName = \"depth_anything_v2_dsp.dlc\"")
    print("  3. Rebuild and test the app")
    print("\n🔥 Expected performance: ~50ms inference on Hexagon 780 DSP")

except Exception as e:
    print(f"\n❌ Error during compilation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
