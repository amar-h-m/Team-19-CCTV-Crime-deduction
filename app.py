import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.video import r3d_18
import numpy as np

# ---------------- UI ----------------
st.set_page_config(page_title="CCTV Crime Detection", layout="wide")
st.markdown("<h1 style='text-align:center;'>🚨 CCTV Crime Detection System</h1>", unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Device: {device}")

# ---------------- PARAMETERS ----------------
SEQUENCE_LEN = 16   # reduced for faster detection
S1_TH = 0.005
S2_TH = 0.15
S3_VOTE_TH = 0.01
VOTE_COUNT = 1

# ---------------- TRANSFORM ----------------
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():

    class Stage1(nn.Module):
        def __init__(self):
            super().__init__()
            m = models.mobilenet_v2(weights=None)
            m.classifier = nn.Identity()
            self.backbone = m
            self.head = nn.Linear(1280, 2)

        def forward(self, x):
            return self.head(self.backbone(x))

    s1 = Stage1().to(device)
    s1.load_state_dict(torch.load(r"C:\Users\amarh\Desktop\My Professionals\Projects\CCTV Crime\Models\stage1_epoch_3.pth", map_location=device))
    s1.eval()

    class Stage2(nn.Module):
        def __init__(self):
            super().__init__()
            base = models.resnet18(weights=None)
            self.cnn = nn.Sequential(*list(base.children())[:-1])
            self.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 1))

        def forward(self, x):
            B,T,C,H,W = x.shape
            feats = []
            for t in range(T):
                feats.append(self.cnn(x[:,t]).view(B,-1))
            return self.fc(torch.stack(feats,1).mean(1))

    s2 = Stage2().to(device)
    s2.load_state_dict(torch.load(r"C:\Users\amarh\Desktop\My Professionals\Projects\CCTV Crime\Models\stage2_rwf_epoch_8.pth", map_location=device))
    s2.eval()

    s3 = r3d_18(pretrained=False)
    s3.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(s3.fc.in_features,1))
    s3.load_state_dict(torch.load(r"C:\Users\amarh\Desktop\My Professionals\Projects\CCTV Crime\Models\s3_r3d_epoch_4.pth", map_location=device))
    s3 = s3.to(device)
    s3.eval()

    return s1, s2, s3

stage1, stage2, stage3 = load_models()

# ---------------- SESSION ----------------
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

# ---------------- TABS ----------------
tab_cam, tab_manual = st.tabs(["📡 Camera", "📂 Manual Test"])

# ================= CAMERA =================
with tab_cam:

    st.subheader("Live Camera")
    run = st.checkbox("Start Camera")

    status_box = st.empty()
    debug_box = st.empty()

    col1, col2, col3 = st.columns([1,2,1])
    frame_box = col2.empty()

    if run:

        cap = cv2.VideoCapture(0)
        clip = []
        s3_scores = []

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224,224))
            frame_box.image(frame, channels="BGR")

            clip.append(frame)

            if len(clip) < SEQUENCE_LEN:
                continue

            # -------- S1 --------
            img = tf(clip[-1]).unsqueeze(0).to(device)
            s1_prob = torch.softmax(stage1(img),1)[0,1].item()

            # -------- S2 --------
            clip_t = torch.stack([tf(f) for f in clip]).unsqueeze(0).to(device)
            s2_prob = torch.sigmoid(stage2(clip_t)).item()

            # -------- S3 --------
            clip3 = clip_t.permute(0,2,1,3,4)
            clip3 = F.interpolate(clip3, size=(32,128,128))
            s3_prob = torch.sigmoid(stage3(clip3)).item()

            # voting
            s3_scores.append(s3_prob)
            if len(s3_scores) > 5:
                s3_scores.pop(0)

            votes = sum(p > S3_VOTE_TH for p in s3_scores)

            # decision
            if (s2_prob > S2_TH and votes >= VOTE_COUNT):
                status_box.error(f"🚨 CRIME ({s3_prob:.3f})")
            else:
                status_box.success(f"Normal ({s3_prob:.3f})")

            # debug info
            debug_box.write({
                "S1": round(s1_prob,3),
                "S2": round(s2_prob,3),
                "S3": round(s3_prob,3),
                "votes": votes
            })

            # sliding window (IMPORTANT FIX)
            clip.pop(0)

        cap.release()

# ================= MANUAL =================
with tab_manual:

    st.subheader("📂 Upload Video for Testing")
    uploaded_file = st.file_uploader("Upload CCTV Video", type=["mp4","avi","mov"])

    if uploaded_file is not None:

        st.video(uploaded_file)

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        if st.button("▶ Run Detection"):
            st.session_state.run_detection = True

        if st.session_state.run_detection:

            cap = cv2.VideoCapture("temp_video.mp4")

            frames = []
            s3_scores = []
            detected = False

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)
            debug_box = st.empty()

            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                count += 1
                progress.progress(count/(total_frames+1))

                frame = cv2.resize(frame,(224,224))
                frames.append(frame)

                if len(frames) == SEQUENCE_LEN:

                    img = tf(frames[-1]).unsqueeze(0).to(device)
                    s1_prob = torch.softmax(stage1(img),1)[0,1].item()

                    clip_t = torch.stack([tf(f) for f in frames]).unsqueeze(0).to(device)
                    s2_prob = torch.sigmoid(stage2(clip_t)).item()

                    clip3 = clip_t.permute(0,2,1,3,4)
                    clip3 = F.interpolate(clip3, size=(32,128,128))
                    s3_prob = torch.sigmoid(stage3(clip3)).item()

                    s3_scores.append(s3_prob)
                    if len(s3_scores) > 5:
                        s3_scores.pop(0)

                    votes = sum(p > S3_VOTE_TH for p in s3_scores)

                    if (s2_prob > S2_TH and votes >= VOTE_COUNT):
                        detected = True

                    debug_box.write({
                        "S1": round(s1_prob,3),
                        "S2": round(s2_prob,3),
                        "S3": round(s3_prob,3),
                        "votes": votes
                    })

                    # sliding window
                    frames.pop(0)

            cap.release()

            st.subheader("🔍 Final Result")

            if detected:
                st.error("🚨 CRIME DETECTED")
            else:
                st.success("✅ NO CRIME")