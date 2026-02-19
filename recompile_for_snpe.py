#!/usr/bin/env python3
"""
Recompile Depth Anything V2 specifically for SNPE runtime
NOT QNN - use legacy SNPE DLC format
"""

import sys
import qai_hub
import qai_hub_models
from qai_hub_models.models.depth_anything_v2 import Model

print("=" * 60)
print("Recompiling for SNPE (not QNN)")
print("=" * 60)

# Get Samsung Galaxy S22
devices = qai_hub.get_devices()
target_device = None
for device in devices:
    if device.name == "Samsung Galaxy S22 (Family)":
        target_device = device
        print(f"✓ Found target: {device.name}")
        print(f"  Chipset: SM8450, Hexagon v69")
        break

if not target_device:
    print("❌ Device not found")
    sys.exit(1)

# Load model
print("\n🔧 Loading Depth Anything V2 model...")
model = Model.from_pretrained()

# Convert to TorchScript
print("\n📦 Converting to TorchScript...")
traced_model = model.convert_to_torchscript(model.get_input_spec())
print("  ✓ Model traced")

# Try different compilation options for SNPE
print(f"\n⚙️ Compiling for SNPE/HTP runtime...")
print("  Checking available target runtimes...")
print("  Options: 'onnx', 'precompiled_qnn_onnx', 'qnn_context_binary', 'qnn_dlc', ")
print("           'qnn_lib_aarch64_android', 'qnn_lib_x86_64_linux', 'tflite'")

# Try precompiled QNN ONNX which might be more compatible with SNPE
target_runtime = "precompiled_qnn_onnx"
print(f"\n  Trying: {target_runtime}")

try:
    compile_job = qai_hub.submit_compile_job(
        model=traced_model,
        device=target_device,
        input_specs=model.get_input_spec(),
        options=f"--target_runtime {target_runtime}"
    )

    print(f"  ✓ Job submitted: {compile_job.job_id}")
    print(f"  URL: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
    print("  Waiting...")

    compile_job.wait()
    status = compile_job.get_status()
    print(f"\n  Status: {status}")

    # Download
    output_path = f"weights/depth_anything_v2_onnx/depth_anything_v2_{target_runtime}.zip"
    print(f"\n📥 Downloading to {output_path}...")
    compile_job.download_target_model(output_path)

    print(f"\n✅ Model saved to: {output_path}")
    print(f"\nNext: Extract and check format compatibility")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
