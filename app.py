"""
😊 Facial Emotion Recognition — Live Camera App
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

# ─── Page Config ───
st.set_page_config(page_title="😊 Face Emotion AI", page_icon="😊", layout="wide")

# ─── Constants ───
CLASSES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
NUM_CLASSES = 7
LABEL_MAP = {"1": "Surprise", "2": "Fear", "3": "Disgust", "4": "Happiness",
             "5": "Sadness", "6": "Anger", "7": "Neutral"}
EMOJI_MAP = {"Surprise": "😲", "Fear": "😨", "Disgust": "🤢",
             "Happiness": "😊", "Sadness": "😢", "Anger": "😠", "Neutral": "😐"}
COLOR_MAP = {"Surprise": (255, 215, 0), "Fear": (148, 0, 211), "Disgust": (0, 128, 0),
             "Happiness": (0, 255, 127), "Sadness": (65, 105, 225), "Anger": (255, 0, 0),
             "Neutral": (200, 200, 200)}
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Definitions ───
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
    def forward(self, x):
        return self.classifier(self.features(x))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
    def forward(self, x):
        return F.relu(self.net(x) + x)

class DeepCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64),
                                    nn.MaxPool2d(2), nn.Dropout2d(0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128),
            ResBlock(128), ResBlock(128),
            nn.MaxPool2d(2), nn.Dropout2d(0.2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes))
    def forward(self, x):
        return self.head(self.layer2(self.layer1(self.stem(x))))

def build_resnet50(num_classes=7):
    model = models.resnet50(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feat, 512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, num_classes))
    return model

def build_efficientnet(num_classes=7):
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    in_feat = model.num_features
    full_model = nn.Sequential(
        model, nn.Dropout(0.4),
        nn.Linear(in_feat, 512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, num_classes))
    return full_model

def build_vit(num_classes=7):
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)
    return model

# ─── Model Loading ───
MODEL_FILES = {
    "SimpleCNN": "rafdb_model_simplecnn.pth",
    "DeepCNN": "rafdb_model_deepcnn.pth",
    "ResNet50": "rafdb_model_resnet50.pth",
    "EfficientNetB3": "rafdb_model_efficientnetb3.pth",
    "ViT-Small": "rafdb_model_vit-small.pth",
}
MODEL_BUILDERS = {
    "SimpleCNN": lambda: SimpleCNN(NUM_CLASSES),
    "DeepCNN": lambda: DeepCNN(NUM_CLASSES),
    "ResNet50": lambda: build_resnet50(NUM_CLASSES),
    "EfficientNetB3": lambda: build_efficientnet(NUM_CLASSES),
    "ViT-Small": lambda: build_vit(NUM_CLASSES),
}

@st.cache_resource
def load_model(name):
    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MODEL_BUILDERS[name]()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

# ─── Transforms ───
small_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
large_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_face(face_img, model_name):
    pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    if model_name in ["SimpleCNN", "DeepCNN"]:
        return small_tf(pil).unsqueeze(0).to(DEVICE)
    return large_tf(pil).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict(model, tensor):
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs[idx], probs

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e, #16213e); }
h1, h2, h3 { color: #e0e0ff !important; }
.emotion-card {
    background: rgba(255,255,255,0.08); border-radius: 16px;
    padding: 20px; text-align: center; backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    transition: transform 0.2s; margin: 8px 0;
}
.emotion-card:hover { transform: translateY(-4px); }
.big-emoji { font-size: 64px; }
.emotion-label { font-size: 24px; font-weight: 800; color: #fff; }
.confidence { font-size: 18px; color: #a0a0ff; }
.prob-bar { height: 8px; border-radius: 4px; margin: 2px 0; }
.status-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 13px; font-weight: 600; color: #fff;
}
.status-on { background: #00c853; }
.status-off { background: #ff1744; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    available = {k: v for k, v in MODEL_FILES.items()
                 if os.path.exists(os.path.join(MODEL_DIR, v))}
    if not available:
        st.error("No model files found!")
        st.stop()
    model_name = st.selectbox("🧠 Select Model", list(available.keys()), index=0)
    conf_thresh = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    face_scale = st.slider("🔍 Face Detection Scale", 1.05, 1.5, 1.3, 0.05)
    min_neighbors = st.slider("👥 Min Neighbors", 1, 10, 5)
    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown(f"**Models found:** {len(available)}")
    for k in available:
        st.markdown(f"- ✅ {k}")

# ─── Header ───
st.markdown("<h1 style='text-align:center;'>😊 Real-Time Face Emotion Recognition</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:#aaa;'>Model: <b>{model_name}</b> | Device: <b>{DEVICE}</b></p>", unsafe_allow_html=True)

# ─── Load Model ───
with st.spinner(f"Loading {model_name}..."):
    model = load_model(model_name)
if model is None:
    st.error(f"Could not load {model_name}!")
    st.stop()
face_cascade = load_face_detector()

# ─── Main Layout ───
col_cam, col_info = st.columns([3, 2])

with col_cam:
    st.markdown("### 📹 Camera Feed")
    run = st.toggle("▶️ Start Camera", value=False)
    frame_holder = st.empty()

with col_info:
    st.markdown("### 📊 Detection Results")
    result_holder = st.empty()
    chart_holder = st.empty()

# ─── Camera Loop ───
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open camera! Check permissions.")
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera read failed.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=face_scale,
                                                   minNeighbors=min_neighbors,
                                                   minSize=(48, 48))
            results = []
            for (x, y, w, h) in faces:
                pad = int(0.1 * w)
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                tensor = preprocess_face(face_crop, model_name)
                emotion, conf, probs = predict(model, tensor)
                results.append({"emotion": emotion, "conf": conf, "probs": probs,
                                "box": (x, y, w, h)})
                color = COLOR_MAP.get(emotion, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{EMOJI_MAP[emotion]} {emotion} {conf:.0%}"
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_holder.image(display, channels="RGB", use_container_width=True)

            # Update results panel
            if results:
                top = max(results, key=lambda r: r["conf"])
                emoji = EMOJI_MAP[top["emotion"]]
                # Use string joining or no-indent f-strings to prevent markdown code block issues
                html = (
                    f"<div class='emotion-card'>"
                    f"<div class='big-emoji'>{emoji}</div>"
                    f"<div class='emotion-label'>{top['emotion']}</div>"
                    f"<div class='confidence'>{top['conf']:.1%} confidence</div>"
                    f"</div>"
                )
                for i, cls in enumerate(CLASSES):
                    p = top["probs"][i]
                    c = COLOR_MAP[cls]
                    hex_c = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
                    html += (
                        f"<div style='display:flex;align-items:center;margin:4px 0;'>"
                        f"<span style='width:90px;color:#ccc;font-size:13px;'>{EMOJI_MAP[cls]} {cls}</span>"
                        f"<div style='flex:1;background:rgba(255,255,255,0.1);border-radius:4px;height:8px;'>"
                        f"<div class='prob-bar' style='width:{p*100:.1f}%;background:{hex_c};'></div>"
                        f"</div>"
                        f"<span style='width:50px;text-align:right;color:#aaa;font-size:12px;'>{p:.1%}</span>"
                        f"</div>"
                    )
                result_holder.write(html, unsafe_allow_html=True)
            else:
                result_holder.markdown(
                    "<div class='emotion-card'><div class='big-emoji'>🔍</div>"
                    "<div class='emotion-label'>No Face Detected</div></div>",
                    unsafe_allow_html=True)
            time.sleep(0.03)
    finally:
        cap.release()
else:
    frame_holder.markdown(
        "<div class='emotion-card' style='height:300px;display:flex;align-items:center;justify-content:center;'>"
        "<div><div class='big-emoji'>📹</div><div class='emotion-label'>Camera Off</div>"
        "<p style='color:#888;'>Toggle the switch above to start</p></div></div>",
        unsafe_allow_html=True)
    result_holder.markdown(
        "<div class='emotion-card'><div class='big-emoji'>⏳</div>"
        "<div class='emotion-label'>Waiting...</div></div>",
        unsafe_allow_html=True)
