# Blender import
import bpy
import bmesh

from pathlib import Path
from uuid import uuid4

from PIL import Image
import matplotlib.pyplot as plt

base_dir = Path("__file__").parent
save_dir = base_dir / "img"
save_path = save_dir / f"render-{str(uuid4())[:8]}.png"
save_path = str(save_path)


bpy.data.scenes[0].render.engine = "CYCLES"

# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

# Set the device and feature set
bpy.context.scene.cycles.device = "GPU"


bpy.ops.wm.open_mainfile(filepath=str(base_dir / "example_blender" / "flat-archiviz.blend"))

for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d.use = True

    if d.type == "CPU":
        d.use = False

    print(f"Device: {d.name}, Use: {d.use}")


for scene in bpy.data.scenes:
    bpy.ops.render.render(write_still=True)
    bpy.data.images["Render Result"].save_render(save_path)


im = Image.open(save_path)
im.show()