"""
😊 Facial Emotion Recognition — Live Camera App (Stable Version)
Uses RAF-DB trained models: SimpleCNN, DeepCNN, ResNet50, EfficientNetB3, ViT-Small
"""
import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm
import os
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# ─── Page Config ───
st.set_page_config(page_title="😊 Face Emotion AI", page_icon="😊", layout="wide")

# ─── Constants ───
CLASSES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
EMOJI_MAP = {"Surprise": "😲", "Fear": "😨", "Disgust": "🤢", "Happiness": "😊", "Sadness": "😢", "Anger": "😠", "Neutral": "😐"}
COLOR_MAP = {"Surprise": (255, 215, 0), "Fear": (148, 0, 211), "Disgust": (0, 128, 0), "Happiness": (0, 255, 127), "Sadness": (65, 105, 225), "Anger": (255, 0, 0), "Neutral": (200, 200, 200)}
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ─── Model Definitions (Omitted for brevity in this block, but preserved in file) ───
# [Keep all class definitions: SimpleCNN, ResBlock, DeepCNN, build_resnet50, build_efficientnet, build_vit]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
    def forward(self, x): return F.relu(self.net(x) + x)

class DeepCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64), nn.MaxPool2d(2), nn.Dropout2d(0.2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128), ResBlock(128), ResBlock(128), nn.MaxPool2d(2), nn.Dropout2d(0.2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, num_classes))
    def forward(self, x): return self.head(self.layer2(self.layer1(self.stem(x))))

def build_resnet50(num_classes=7):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, num_classes))
    return model

def build_efficientnet(num_classes=7):
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    return nn.Sequential(model, nn.Dropout(0.4), nn.Linear(model.num_features, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, num_classes))

def build_vit(num_classes=7): return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)

# ─── Loading Logic ───
MODEL_FILES = {"SimpleCNN": "rafdb_model_simplecnn.pth", "DeepCNN": "rafdb_model_deepcnn.pth", "ResNet50": "rafdb_model_resnet50.pth", "EfficientNetB3": "rafdb_model_efficientnetb3.pth", "ViT-Small": "rafdb_model_vit-small.pth"}
MODEL_BUILDERS = {"SimpleCNN": lambda: SimpleCNN(7), "DeepCNN": lambda: DeepCNN(7), "ResNet50": lambda: build_resnet50(7), "EfficientNetB3": lambda: build_efficientnet(7), "ViT-Small": lambda: build_vit(7)}

@st.cache_resource
def load_model(name):
    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    if not os.path.exists(path): return None
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MODEL_BUILDERS[name]()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(DEVICE).eval()

@st.cache_resource
def load_face_detector(): return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

small_tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((48, 48)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
large_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def preprocess_face(face_img, model_name):
    pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return small_tf(pil).unsqueeze(0).to(DEVICE) if model_name in ["SimpleCNN", "DeepCNN"] else large_tf(pil).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict(model, tensor):
    probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs[idx], probs

# ─── Sidebar ───
with st.sidebar:
    st.title("🎛️ AI Settings")
    model_name = st.selectbox("🧠 Model", list(MODEL_FILES.keys()))
    face_scale = st.slider("🔍 Face Scale", 1.05, 1.5, 1.3)
    min_neighbors = st.slider("👥 Min Neighbors", 1, 10, 5)

# ─── Video Processor ───
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = load_face_detector()
        self.model = load_model(model_name)
        self.model_name = model_name

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, face_scale, min_neighbors, minSize=(48,48))
        for (x, y, w, h) in faces:
            crop = img[max(0,y-20):min(img.shape[0],y+h+20), max(0,x-20):min(img.shape[1],x+w+20)]
            if crop.size > 0:
                emotion, conf, _ = predict(self.model, preprocess_face(crop, self.model_name))
                cv2.rectangle(img, (x, y), (x+w, y+h), COLOR_MAP.get(emotion, (255,255,255)), 2)
                cv2.putText(img, f"{emotion} {conf:.0%}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_MAP.get(emotion, (255,255,255)), 2)
        import av
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─── UI ───
st.markdown("<h1 style='text-align:center;'>😊 Real-Time Face Emotion AI</h1>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📹 Live Camera", "📤 Upload Image"])

with tab1:
    st.info("💡 Click START to activate your camera. Make sure your browser has permission.")
    webrtc_streamer(key="face-ai", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIG, 
                    video_processor_factory=EmotionProcessor, media_stream_constraints={"video": True, "audio": False})

with tab2:
    f = st.file_uploader("Upload a face photo", type=["jpg", "png"])
    if f:
        img = np.array(Image.open(f).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = load_face_detector().detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop = img[max(0,y-20):y+h+20, max(0,x-20):x+w+20]
            emotion, conf, probs = predict(load_model(model_name), preprocess_face(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), model_name))
            st.write(f"### {EMOJI_MAP[emotion]} Predicted: {emotion} ({conf:.1%})")
            st.image(crop, width=150)
            for i, cls in enumerate(CLASSES):
                st.progress(float(probs[i]), text=f"{cls}: {probs[i]:.1%}")
