#!/usr/bin/env python3
"""
Check details of the compiled model job
"""

import qai_hub

job_id = "j5mzoev9p"

print(f"Retrieving job {job_id} details...")
job = qai_hub.get_job(job_id)

print(f"\nJob ID: {job.job_id}")
print(f"Status: {job.get_status()}")
print(f"Job type: {job.job_type}")

# Get target device info
if hasattr(job, 'device'):
    print(f"Target device: {job.device}")

# Get compile options
if hasattr(job, 'options'):
    print(f"Compile options: {job.options}")

# Get all available attributes
print(f"\nAll job attributes:")
for attr in dir(job):
    if not attr.startswith('_'):
        try:
            value = getattr(job, attr)
            if not callable(value):
                print(f"  {attr}: {value}")
        except:
            pass
