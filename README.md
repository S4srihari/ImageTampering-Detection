# Digital Image Tampering (Deep Fakes or Tamperings) Detection using ELA and CNN.

### Overview  
This project detects **digital image forgeries** using a combination of **Error Level Analysis (ELA)** and **Convolutional Neural Networks (CNNs)**.  
With image manipulation becoming easier through tools like Photoshop or GIMP, this model provides an **automated, deep learning–based approach** to verify image authenticity.  

The application:
-> Takes a **JPEG image** as input  
-> Performs **Error Level Analysis (ELA)** to highlight tampered regions  
-> Classifies the image as **“Authentic”** or **“Tampered”**  
-> Optionally visualizes the **tampered areas** through ELA intensity maps  


## Features
-> **CNN-based Classification:** Custom-built model trained from scratch using TensorFlow/Keras.  
-> **Error Level Analysis (ELA):** Reveals hidden compression inconsistencies caused by tampering.  
-> **High Accuracy:** Achieved ~92% accuracy on a balanced tampering dataset.  
-> **Visual Explainability:** Displays manipulated areas using enhanced ELA heatmaps.  
-> **Fully Automated Pipeline:** From preprocessing to prediction.  


## Project Structure

Digital-Image-Tampering-Detection/
│
├── main.ipynb     # Jupyter Notebook containing full pipeline
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies


## Installation

### 1. Clone this Repository
bash
git clone https://github.com/<your-username>/Digital-Image-Tampering-Detection.git
cd Digital-Image-Tampering-Detection```

### 2. Create Virtual Environment
bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows


### 3. Install Dependencies
bash
pip install -r requirements.txt

**requirements.txt**
tensorflow==2.11.0
keras==2.11.0
pillow
numpy
matplotlib
scikit-learn


## How It Works

1. **Error Level Analysis (ELA):**  
   - The input JPEG image is re-saved at 90% quality.  
   - The recompressed image is subtracted from the original to reveal pixel-level differences.  
   - Tampered regions exhibit stronger ELA responses (brighter spots).  

2. **CNN Classification:**  
   - A 2-layer CNN is trained on ELA-transformed images.  
   - The network outputs probabilities for **“Authentic”** or **“Tampered.”**

3. **Visualization:**  
   - If an image is flagged as tampered, the ELA map highlights potential manipulation zones.


## Usage

### Run the Jupyter Notebook
Open "sriharip_cvip_project.ipynb" and execute all cells:
bash
jupyter notebook sriharip_cvip_project.ipynb


Or, to predict a single image:
python
from ELA_CNN_Detector import predict_image

result, ela_image = predict_image("test_image.jpg")
print("Prediction:", result)
ela_image.show()




## Results

| Dataset | Accuracy | Precision |
|----------|-----------|------------|
| Validation | 90.5% | 95.1% |
| Test | 90.7% | 95.6% |



## Dataset
The model was trained on the **[CASIA Image Tampering Dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset)**.  
It includes both *authentic* and *tampered* images featuring splicing, cloning, and object removal.


## Future Enhancements
-> Add segmentation-based tamper localization (U-Net).  
-> Deploy as a web application using Flask/Django.  
->  Support other image formats (PNG, TIFF).  
-> Extend to multi-class classification (splicing, cloning, erasure).  


## References
-> H. Farid, *“Exposing Digital Forgeries from JPEG Ghosts,”* IEEE TIFS, 2009.  
-> [Keras Documentation](https://keras.io)  
-> [Pillow Documentation](https://pillow.readthedocs.io)  
-> [CASIA Dataset on Kaggle](https://www.kaggle.com/datasets/sophatvathana/casia-dataset)
