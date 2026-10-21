from pathlib import Path
from PIL import Image
from lwcc import LWCC

image_path = Path(r"D:\jhu_crowd_v2.0\train\images\0018.jpg")

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

with Image.open(image_path) as image:
    print("Image size:", image.size)

print("\nAvailable LWCC methods:")
print([name for name in dir(LWCC) if not name.startswith("_")])

print("\nRunning CSRNet...")
result = LWCC.get_count(
    str(image_path),
    model_name="CSRNet",
    model_weights="SHA",
)

print("\nOutput type:", type(result))
print("Output value:", result)