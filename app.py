from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from gfpgan import GFPGANer
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load GFPGAN model once
restorer = GFPGANer(
    model_path='C:/Users/ASUS/Downloads/GFPGANv1.4.pth',
    upscale=2, 
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

@app.route('/', methods=['GET', 'POST'])
def index():
    restored_img_path = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = file.filename
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            output_filename = 'restored_' + filename
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            file.save(input_path)
            img = cv2.imread(input_path)
            if img is None:
                print('Error: Uploaded image could not be read!')
                return render_template('index.html', restored_img_path=None)
            
            # Enhance image using GFPGAN
            result = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            if isinstance(result, tuple):
                restored_img = result[1]
            else:
                restored_img = result

            if isinstance(restored_img, list):
                restored_img = np.array(restored_img)
                
            if isinstance(restored_img, np.ndarray):
                restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)
                if len(restored_img.shape) == 4:
                    restored_img = restored_img[0]
                cv2.imwrite(output_path, restored_img)
                restored_img_path = 'output/' + output_filename

    return render_template('index.html', restored_img_path=restored_img_path)

if __name__ == '__main__':
    app.run(debug=True)
