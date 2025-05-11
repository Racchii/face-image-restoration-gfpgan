import cv2

image_path = r"C:/Users/ASUS\Desktop/low_quality1.jpg"  # Ensure the path is correct
img = cv2.imread(image_path)

if img is None:
    print("Error: Image could not be loaded!")
else:
    print("Image loaded successfully!")
    print(f"Image shape: {img.shape}")
