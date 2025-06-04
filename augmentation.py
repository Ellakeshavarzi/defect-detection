import cv2
import os
import albumentations as A

# === CONFIGURATION ===
input_dir = "brick_tiled"        # Folder with your original 96 preprocessed images
output_dir = "brick_augmented"          # Folder to store augmented images
augmentations_per_image = 5             # Number of unique augmented versions per image
tile_size = 150                         # Image size
os.makedirs(output_dir, exist_ok=True)

# === DEFINE AUGMENTATION PIPELINE ===
transform = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
    ], p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=25, p=0.7),
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    # A.MotionBlur(p=0.3),
    A.RandomCrop(width=tile_size, height=tile_size, p=0.5)
])

# === AUGMENT ALL IMAGES IN THE FOLDER ===
for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Failed to load image: {filename}")
        continue

    base_name = os.path.splitext(filename)[0]

    for i in range(augmentations_per_image):
        augmented = transform(image=image)["image"]
        new_name = f"{base_name}_aug_{i:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, new_name), augmented)

    print(f"✅ {filename} → {augmentations_per_image} augmentations")

print("\n🎉 All images have been augmented and saved to:", output_dir)
