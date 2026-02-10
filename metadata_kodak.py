import os
from PIL import Image

def inspect_png_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg"):
            path = os.path.join(directory, filename)

            with Image.open(path) as img:
                width, height = img.size

                print("=" * 50)
                print(f"File: {filename}")
                print(f"Format: {img.format}")
                print(f"Mode: {img.mode}")
                print(f"Size (pixels): {width} x {height}")

                # Optional: print metadata if it exists
                if img.info:
                    print("Metadata:")
                    for k, v in img.info.items():
                        print(f"  {k}: {v}")
                else:
                    print("Metadata: None")

#inspect_png_images("/home/anas/datasets/kodak")
inspect_png_images("/home/anas/thesis/openimages/scissors/images")

