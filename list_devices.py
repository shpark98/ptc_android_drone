#!/usr/bin/env python3
"""
List all available devices in Qualcomm AI Hub and find Snapdragon 8 Gen 1 devices
"""

import qai_hub

print("Fetching devices from Qualcomm AI Hub...")
devices = qai_hub.get_devices()

print(f"\nFound {len(devices)} devices\n")
print("Searching for Snapdragon 8 Gen 1 / SM8450 / Hexagon 780 devices...")
print("=" * 80)

sd8gen1_devices = []
for device in devices:
    attrs = device.attributes if hasattr(device, 'attributes') else []
    attrs_str = str(attrs).lower()

    # Look for SD8Gen1, SM8450, or related Hexagon versions
    if any(keyword in device.name.lower() or keyword in attrs_str for keyword in ['8gen1', '8 gen 1', 'sm8450', 'v73', 'hexagon:v73']):
        sd8gen1_devices.append(device)
        print(f"\nDevice: {device.name}")
        print(f"  OS: {device.os}")
        if hasattr(device, 'attributes'):
            print(f"  Attributes:")
            for attr in device.attributes:
                if 'hexagon' in attr.lower() or 'chipset' in attr.lower() or 'soc' in attr.lower():
                    print(f"    - {attr}")

if sd8gen1_devices:
    print(f"\n\n✓ Found {len(sd8gen1_devices)} matching devices")
else:
    print("\n\n⚠️ No exact matches found. Listing all Samsung S22 devices:")
    for device in devices:
        if 's22' in device.name.lower() or 'samsung galaxy s22' in device.name.lower():
            print(f"\nDevice: {device.name}")
            print(f"  OS: {device.os}")
            if hasattr(device, 'attributes'):
                print(f"  Attributes: {device.attributes}")
