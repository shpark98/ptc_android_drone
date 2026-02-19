#!/usr/bin/env python3
"""
Download the successfully compiled DSP model from Qualcomm AI Hub
Job ID: j5mzoev9p
"""

import sys
import qai_hub

print("=" * 60)
print("Downloading Compiled DSP Model from Qualcomm AI Hub")
print("=" * 60)

# Job ID from the successful compilation
job_id = "j5mzoev9p"

print(f"\n📥 Retrieving job {job_id}...")

try:
    # Get the compile job
    compile_job = qai_hub.get_job(job_id)

    print(f"  Job status: {compile_job.get_status()}")
    print(f"  Job URL: https://workbench.aihub.qualcomm.com/jobs/{job_id}/")

    # Download the compiled model
    output_path = "weights/depth_anything_v2_onnx/depth_anything_v2_dsp_htp.dlc"
    print(f"\n📥 Downloading compiled DLC to {output_path}...")

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
    print(f"\n❌ Error downloading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
