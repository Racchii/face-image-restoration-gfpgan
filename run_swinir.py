import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
import numpy as np
import os

# Paths
model_path = 'C:/Users/ASUS/Downloads/GFPGANv1.4.pth'  # Path to your GFPGAN model
input_path = 'C:/Users/ASUS/Downloads/images/low_quality1.jpg'  # Path to your low-quality input image
output_path = 'C:/Users/ASUS/Desktop/SwinIR/output/output.png'  # Try PNG format

# Ensure the output folder exists
output_folder = os.path.dirname(output_path)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Output folder created: {output_folder}")

# Test model loading
try:
    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)

# Read input image
img = cv2.imread(input_path)
if img is None:
    raise Exception("Image not found at the given path!")

# Restore faces
result = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

# Print the result of enhance() to understand its structure
print("Enhance result:", result)

# Extract the restored image from the tuple
if isinstance(result, tuple):
    print("Result is a tuple. Extracting the restored image.")
    restored_img = result[1]  # The second item in the tuple is the restored image
else:
    restored_img = result  # If result is not a tuple, assume it's the image

# If the restored image is still a list, convert it to a NumPy array
if isinstance(restored_img, list):
    print("Restored image is a list. Converting to NumPy array.")
    restored_img = np.array(restored_img)

# Remove unnecessary dimensions (1, 512, 512, 3) => (512, 512, 3)
restored_img = np.squeeze(restored_img)

# Print the type and shape of the restored image
print("Restored image type:", type(restored_img))
if isinstance(restored_img, np.ndarray):
    print("Restored image shape:", restored_img.shape)
    print("Restored image dtype:", restored_img.dtype)
else:
    print("Restored image is not a valid NumPy array.")

# Ensure the restored image is in the correct format for saving (uint8)
if isinstance(restored_img, np.ndarray):
    restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)

    # Try saving the image directly as PNG
    try:
        # Try saving as PNG format
        if cv2.imwrite(output_path, restored_img):
            print(f"✅ Image saved successfully at {output_path}")
        else:
            print(f"❌ Error saving image at {output_path}")
    except Exception as e:
        print(f"Error occurred during saving the image: {e}")
else:
    print("Restored image is not in the expected format.")
