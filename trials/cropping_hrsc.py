from pathlib import Path
from PIL import Image

# ===== CONFIG =====
INPUT_DIR = Path("/run/media/anasnamouchi/OS/datasets/full_hrsc2016")
OUTPUT_DIR = Path("/run/media/anasnamouchi/OS/datasets/cropped_hrsc2016")
DIVISOR = 64
# ==================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def center_crop_to_divisible(img: Image.Image, divisor: int) -> Image.Image:
    w, h = img.size

    new_w = (w // divisor) * divisor
    new_h = (h // divisor) * divisor

    if new_w == w and new_h == h:
        return img.copy()

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    return img.crop((left, top, right, bottom))


def process_image(path: Path):
    img = Image.open(path).convert("RGB")

    cropped = center_crop_to_divisible(img, DIVISOR)

    out_path = OUTPUT_DIR / path.name
    cropped.save(out_path, format="BMP")

    print(f"{path.name}: {img.size} -> {cropped.size}")


def main():
    for img_path in INPUT_DIR.glob("*.bmp"):#
        try:
            process_image(img_path)
        except Exception as e:
            print(f"Exception: {Exception}")

if __name__ == "__main__":
    main()
