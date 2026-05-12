# 😊 Facial Emotion Recognition with Streamlit

A real-time facial emotion recognition application built with Streamlit, OpenCV, and PyTorch. The models are trained on the **RAF-DB** dataset.

## 🚀 Features
- **Live Camera Feed:** Real-time detection and classification of facial emotions.
- **Multiple Model Architectures:**
  - Simple CNN
  - Deep CNN (with Residual Blocks)
  - ResNet50
  - EfficientNetB3
  - Vision Transformer (ViT-Small)
- **Interactive UI:** Beautiful dashboard with probability bars and confidence metrics.
- **Interactive Face Detection:** Adjustable detection parameters (Scale Factor, Min Neighbors).

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ahsitab/Facial-Emotion-Recognition.git
   cd Facial-Emotion-Recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset
The models were trained on the **RAF-DB (Real-world Affective Faces Database)**, which includes 7 basic emotions:
1. Surprise
2. Fear
3. Disgust
4. Happiness
5. Sadness
6. Anger
7. Neutral

## 🧠 Models
The project includes a variety of models, from simple custom CNNs to state-of-the-art architectures like Vision Transformers and EfficientNet.

## 📜 License
This project is for educational purposes.
