# Facial Image Restoration using GFPGAN

This project is a **Facial Image Restoration System** built using **GFPGAN (Generative Facial Prior-GAN)**.  
It enhances old, blurred, low-resolution, or damaged facial images using a pre-trained deep learning model while preserving natural facial details.

The system provides a clean and interactive **web interface using Flask**, allowing users to upload images and obtain restored results in seconds.

---

## 🚀 Features

- ✔ Real-time facial restoration using GFPGAN
- ✔ Enhances resolution, sharpness & facial details
- ✔ Supports multiple image formats (PNG, JPG, JPEG)
- ✔ Simple & responsive Flask UI
- ✔ Works locally (offline) after setup
- ✔ No complex ML knowledge required to use
- ✔ Fast inference on GPU (optional) and CPU-supported with slower processing

---

## 🧠 Model Overview — GFPGAN

**GFPGAN** (Generative Facial Prior-GAN) is a state-of-the-art GAN model trained on a massive facial dataset using facial priors for:
- Super-resolution
- Deblurring
- Restoration
- Detail enhancement

Paper: *GFPGAN: Towards Real-World Blind Face Restoration with Generative Facial Prior*

Repo (Original): https://github.com/TencentARC/GFPGAN *(credit given to authors)*

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Model | GFPGAN |
| Backend | Flask (Python) |
| Frontend | HTML + CSS |
| DL Framework | PyTorch |
| Deployment | Localhost |
| Input | Single Image Upload |
| Output | Restored Image |

---

## 📂 Project Structure
📁 Facial-Restoration-Project
├── app.py # Flask backend
├── static/
│ └── restored/ # Output saved images
├── templates/
│ └── index.html # UI frontend
├── gfpgan/
│ ├── weights/... # Pretrained model weights
│ ├── inference_code.py # restoration logic
└── README.md



---

## ⚙ Installation & Setup

 **1. Clone Repository**
git clone https://github.com/yourusername/Facial-Restoration-GFPGAN.git
cd Facial-Restoration-GFPGAN

Create Virtual Environment
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows


Install Dependencies
pip install -r requirements.txt


 Download GFPGAN Weights
Download pretrained model weights from GFPGAN repository and place under:

/gfpgan/weights/



▶ Running the Application

start the Flask app:

python app.py


Visit in browser:

http://127.0.0.1:5000/
Upload an image → Click Restore → View/Download result.

🎯 Applications

Old photo restoration
Low-quality CCTV facial enhancement
Research & forensics
Photo archival
AI-based image enhancement

📜 License & Credits

This project uses the GFPGAN model from Tencent ARC.
Original model credit:
https://github.com/TencentARC/GFPGAN

This repository is for educational & personal research purposes.
