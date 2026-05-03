import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.video import r3d_18
import numpy as np   # ✅ ADDED

# ---------------- UI ----------------
st.set_page_config(page_title="CCTV Crime Detection", layout="wide")
st.markdown("<h1 style='text-align:center;'>🚨 CCTV Crime Detection System</h1>", unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Device: {device}")

# ---------------- STAGE-0 FUNCTION ----------------
def motion_score(frames):   # ✅ ADDED
    diffs = []
    for i in range(len(frames)-1):
        f1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(f1, f2)
        diffs.append(diff.mean())
    return np.mean(diffs)

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

    return s1,s2,s3


stage1, stage2, stage3 = load_models()

# ---------------- TRANSFORM ----------------
tf = transforms.Compose([
    transforms.ToTensor()
])

# ---------------- PARAMETERS ----------------
SEQUENCE_LEN = 32
S1_TH = 0.03
S2_TH = 0.38
S3_TH = 0.4

if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

# ---------------- TABS ----------------
tab_cam, tab_manual = st.tabs(["📡 Camera", "📂 Manual Test"])

# ================= CAMERA =================
with tab_cam:

    st.subheader("Live Camera")

    run = st.checkbox("Start Camera")

    status_box = st.empty()

    col1, col2, col3 = st.columns([1,2,1])
    frame_box = col2.empty()

    if run:

        cap = cv2.VideoCapture(0)

        clip = []
        s3_scores = []

        for _ in range(300):

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (600, 500))
            frame_box.image(frame, channels="BGR")

            clip.append(frame)

            if len(clip) < SEQUENCE_LEN:
                continue

            # ---------------- S0 MOTION ----------------
            motion = motion_score(clip[-SEQUENCE_LEN:])
            if motion < 6:
                status_box.info("S0: No Motion")
                clip.pop(0)
                continue
            # ------------------------------------------

            img = tf(clip[-1]).unsqueeze(0).to(device)
            s1_prob = torch.softmax(stage1(img),1)[0,1].item()

            if s1_prob <= S1_TH:
                status_box.success("S1: Not Detected")

            else:
                clip_t = torch.stack([tf(f) for f in clip]).unsqueeze(0).to(device)
                s2_prob = torch.sigmoid(stage2(clip_t)).item()

                if s2_prob <= S2_TH:
                    status_box.info("S2: Not Detected")

                else:
                    clip3 = clip_t.permute(0,2,1,3,4)
                    clip3 = F.interpolate(clip3, size=(32,128,128))

                    s3_prob = torch.sigmoid(stage3(clip3)).item()

                    s3_scores.append(s3_prob)
                    if len(s3_scores) > 5:
                        s3_scores.pop(0)

                    votes = sum(p > 0.03 for p in s3_scores)

                    if votes >= 3:
                        status_box.error("S3: Detected")
                    else:
                        status_box.warning("S3: Not Detected")

            clip.pop(0)

        cap.release()

# ================= MANUAL TEST =================
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

            s1_flag = False
            s2_flag = False
            s3_flag = False
            s3_scores = []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)

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

                    # ---------------- S0 MOTION ----------------
                    motion = motion_score(frames)
                    if motion < 6:
                        frames = []
                        continue
                    # ------------------------------------------

                    img = tf(frames[-1]).unsqueeze(0).to(device)
                    s1_prob = torch.softmax(stage1(img),1)[0,1].item()

                    if s1_prob > S1_TH:
                        s1_flag = True

                        clip_t = torch.stack([tf(f) for f in frames]).unsqueeze(0).to(device)
                        s2_prob = torch.sigmoid(stage2(clip_t)).item()

                        if s2_prob > S2_TH:
                            s2_flag = True

                            clip3 = clip_t.permute(0,2,1,3,4)
                            clip3 = F.interpolate(clip3, size=(32,128,128))

                            s3_prob = torch.sigmoid(stage3(clip3)).item()

                            s3_scores.append(s3_prob)
                            if len(s3_scores) > 5:
                                s3_scores.pop(0)

                            votes = sum(p > 0.01 for p in s3_scores)

                            if votes >= 3:
                                s3_flag = True

                    frames = []

            cap.release()

            st.subheader("🔍 Final Pipeline Result")

            st.write(f"S1: {'Detected' if s1_flag else 'Not Detected'}")
            st.write(f"S2: {'Detected' if s2_flag else 'Not Detected'}")
            st.write(f"S3: {'Detected' if s3_flag else 'Not Detected'}")

            if s3_flag:
                st.error("🚨 FINAL RESULT: CRIME DETECTED")
            else:
                st.success("✅ FINAL RESULT: NO CRIME")