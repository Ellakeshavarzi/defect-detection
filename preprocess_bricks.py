import cv2
import os
import numpy as np

# === CONFIG ===
input_dir = "brick_images"              # Raw captured images
output_dir = "brick_tiled"              # Folder for output patches
tile_size = (224, 224)                  # Size of each tile (w, h)
stride = 150                            # Use stride < tile size for overlapping tiles
padding = 10                            # Extra margin around detected brick

os.makedirs(output_dir, exist_ok=True)

def extract_tiles(image, tile_w, tile_h, stride):
    tiles = []
    h, w = image.shape[:2]
    for y in range(0, h - tile_h + 1, stride):
        for x in range(0, w - tile_w + 1, stride):
            tile = image[y:y+tile_h, x:x+tile_w]
            tiles.append(tile)
    return tiles

def process_image(file_path, filename):
    image = cv2.imread(file_path)
    if image is None:
        print(f"❌ Failed to load: {filename}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"⚠️ No contours found in {filename}")
        return

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Apply padding
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)

    cropped = image[y:y+h, x:x+w]
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Extract tiles
    tiles = extract_tiles(cropped_gray, tile_size[0], tile_size[1], stride)

    # Save tiles
    for i, tile in enumerate(tiles):
        tile_name = f"{os.path.splitext(filename)[0]}_tile_{i:03d}.jpg"
        tile_path = os.path.join(output_dir, tile_name)
        cv2.imwrite(tile_path, tile)

    print(f"✅ {filename} → {len(tiles)} tiles saved.")

# Process all images
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        file_path = os.path.join(input_dir, filename)
        process_image(file_path, filename)

print("\n🎉 All brick images processed and tiled!")
