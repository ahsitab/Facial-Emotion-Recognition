"""
😊 Facial Emotion Recognition — Premium Stable App
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
import threading

# ─── Page Config ───
st.set_page_config(page_title="😊 Face Emotion AI", page_icon="😊", layout="wide")

# ─── Constants ───
CLASSES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
EMOJI_MAP = {"Surprise": "😲", "Fear": "😨", "Disgust": "🤢", "Happiness": "😊", "Sadness": "😢", "Anger": "😠", "Neutral": "😐"}
COLOR_MAP = {"Surprise": (255, 215, 0), "Fear": (148, 0, 211), "Disgust": (0, 128, 0), "Happiness": (0, 255, 127), "Sadness": (65, 105, 225), "Anger": (255, 0, 0), "Neutral": (200, 200, 200)}
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
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*6*6, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))
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

# ─── Custom CSS (Restored Premium Style) ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e, #16213e); }
h1, h2, h3 { color: #e0e0ff !important; }
.emotion-card {
    background: rgba(255,255,255,0.1); border-radius: 20px;
    padding: 30px; text-align: center; backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-bottom: 25px;
}
.big-emoji { font-size: 80px; margin-bottom: 10px; }
.emotion-label { font-size: 32px; font-weight: 800; color: #fff; text-transform: uppercase; letter-spacing: 2px; }
.confidence { font-size: 20px; color: #a0a0ff; font-weight: 500; }
.prob-container { margin: 10px 0; }
.prob-label { font-size: 14px; color: #ccc; margin-bottom: 4px; display: flex; justify-content: space-between; }
.prob-bar-bg { background: rgba(255,255,255,0.1); height: 10px; border-radius: 5px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 5px; transition: width 0.5s ease-out; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.title("🎛️ AI Settings")
    model_name = st.selectbox("🧠 Model", list(MODEL_FILES.keys()))
    face_scale = st.slider("🔍 Face Scale", 1.05, 1.5, 1.3)
    min_neighbors = st.slider("👥 Min Neighbors", 1, 10, 5)
    st.markdown("---")
    enable_live = st.checkbox("🧪 Beta: Experimental Live Feed", help="Only enable this if you are on Localhost. Can crash on Cloud.")

face_detector = load_face_detector()
emotion_model = load_model(model_name)

# ─── Helper: Render Emotion Card ───
def render_results(results):
    if not results: return
    top = max(results, key=lambda x: x['conf'])
    st.markdown(f"""
    <div class='emotion-card'>
        <div class='big-emoji'>{EMOJI_MAP[top['emotion']]}</div>
        <div class='emotion-label'>{top['emotion']}</div>
        <div class='confidence'>{top['conf']:.1%} confidence</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Probability Breakdown")
    for i, cls in enumerate(CLASSES):
        p = top['probs'][i]
        c = COLOR_MAP[cls]
        hex_c = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        st.markdown(f"""
        <div class='prob-container'>
            <div class='prob-label'><span>{EMOJI_MAP[cls]} {cls}</span><span>{p:.1%}</span></div>
            <div class='prob-bar-bg'><div class='prob-bar-fill' style='width:{p*100}%; background:{hex_c};'></div></div>
        </div>
        """, unsafe_allow_html=True)

# ─── UI Layout ───
st.markdown("<h1 style='text-align:center;'>😊 Premium Face Emotion AI</h1>", unsafe_allow_html=True)
t1, t2 = st.tabs(["📸 Camera Feed", "📤 Upload Image"])

with t1:
    col_cam, col_res = st.columns([1, 1])
    with col_cam:
        if enable_live:
            # ONLY IMPORT AND RUN WEBRTC IF ENABLED
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
            class EmotionProcessor(VideoProcessorBase):
                def __init__(self):
                    self.detector, self.model = face_detector, emotion_model
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.detector.detectMultiScale(gray, face_scale, min_neighbors)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
                    import av
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV, video_processor_factory=EmotionProcessor, media_stream_constraints={"video": True, "audio": False})
            st.warning("Live results are drawn on the video feed. Switch to 'Stable' mode for the detailed card.")
        else:
            st.info("💡 Snapshot Mode: Take a picture for high-accuracy analysis.")
            img_file = st.camera_input("Capture Emotion")
            if img_file:
                img = np.array(Image.open(img_file).convert("RGB"))
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                faces = face_detector.detectMultiScale(gray, face_scale, min_neighbors)
                if len(faces) > 0:
                    with col_res:
                        results = []
                        for (x, y, w, h) in faces:
                            crop = img[max(0,y-20):y+h+20, max(0,x-20):x+w+20]
                            emotion, conf, probs = predict(emotion_model, preprocess_face(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), model_name))
                            results.append({'emotion': emotion, 'conf': conf, 'probs': probs})
                        render_results(results)
                else: st.warning("No face detected.")

with t2:
    f = st.file_uploader("Upload a photo", type=["jpg", "png"])
    if f:
        img = np.array(Image.open(f).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray, face_scale, min_neighbors)
        if len(faces) > 0:
            col_img, col_res = st.columns([1, 1])
            with col_img: st.image(img, use_container_width=True)
            with col_res:
                results = []
                for (x, y, w, h) in faces:
                    crop = img[max(0,y-20):y+h+20, max(0,x-20):x+w+20]
                    emotion, conf, probs = predict(emotion_model, preprocess_face(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), model_name))
                    results.append({'emotion': emotion, 'conf': conf, 'probs': probs})
                render_results(results)
