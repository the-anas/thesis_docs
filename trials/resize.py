import cv2

INPUT_PATH = "/home/anas/datasets/exp1/urban1.png"
OUTPUT_PATH = "/home/anas/datasets/exp1/urban1_256.png"

img = cv2.imread(INPUT_PATH)
img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(OUTPUT_PATH, img_resized)

print(f"Saved {img.shape[:2]} → {img_resized.shape[:2]}")