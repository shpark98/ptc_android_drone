#!/usr/bin/env python3
"""
Compile Depth Anything V2 for Qualcomm DSP/HTP runtime
Targets: Snapdragon 8 Gen 1 (SM8450)
"""

import os
import sys

# Check if API token is provided
api_token = os.environ.get('QAI_HUB_API_TOKEN')
if not api_token:
    print("=" * 60)
    print("Qualcomm AI Hub - API Token Required")
    print("=" * 60)
    print("\nPlease provide your API token:")
    print("  export QAI_HUB_API_TOKEN='your_token_here'")
    print("  python3 compile_depth_anything_dsp_v2.py")
    print("\nOr create ~/.qai_hub/client.ini with:")
    print("  [default]")
    print("  api_token = your_token_here")
    sys.exit(1)

# Set API token for qai_hub
os.environ['QAI_HUB_API_TOKEN'] = api_token

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
    print(f"  Found {len(devices)} available devices")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
    print("\nPlease check your API token.")
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
    print("\n⚠️ Snapdragon 8 Gen 1 not found. Available devices:")
    for i, device in enumerate(devices[:10]):
        print(f"  {i}: {device.name}")

    choice = input("\nEnter device number to use (or press Enter for first device): ").strip()
    if choice.isdigit():
        target_device = devices[int(choice)]
    else:
        target_device = devices[0]
    print(f"  Using: {target_device.name}")

# Load Depth Anything V2 model
print("\n🔧 Loading Depth Anything V2 model...")
print("  This will download the model from Hugging Face...")
model = Model.from_pretrained()

# Compile for DSP/HTP
print(f"\n⚙️ Compiling for DSP/HTP on {target_device.name}...")
print("  Target: Hexagon HTP (DSP)")
print("  This may take 5-10 minutes...")

try:
    compile_job = qai_hub.submit_compile_job(
        model=model.get_model(),
        device=target_device,
        input_specs=model.get_input_spec(),
        options="--target_runtime htp"  # Force HTP (Hexagon Tensor Processor / DSP)
    )

    print(f"  ✓ Compile job submitted: {compile_job.job_id}")
    print("  Waiting for compilation...")

    # Wait for job to complete
    compile_job.wait()

    if compile_job.is_failed():
        print(f"\n❌ Compilation failed:")
        print(compile_job.get_status())
        sys.exit(1)

    # Download compiled DLC
    print(f"\n✓ Compilation succeeded!")
    output_path = "weights/depth_anything_v2_onnx/depth_anything_v2_dsp_htp.dlc"
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
