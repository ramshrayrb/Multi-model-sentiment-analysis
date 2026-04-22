import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# pip install transformers torch librosa opencv-python
# ─────────────────────────────────────────────

BASE_DIR   = r"D:\multimodel\dataset"
VIDEO_DIR  = os.path.join(BASE_DIR, "video")
AUDIO_DIR  = os.path.join(BASE_DIR, "audio")
TEXT_DIR   = os.path.join(BASE_DIR, "text")
OUTPUT_DIR = os.path.join(BASE_DIR, "features")

os.makedirs(os.path.join(OUTPUT_DIR, "text"),  exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "audio"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "video"), exist_ok=True)


# ═══════════════════════════════════════════════════════
# STEP 1 — TEXT (BERT)
# What it does: reads each .textonly file, passes the
# text through BERT, and saves one 768-dim vector per
# word token.
# Output shape: (seq_len, 768)
# ═══════════════════════════════════════════════════════
def extract_text_features():
    print("\n" + "="*55)
    print("  STEP 1 — TEXT (BERT)")
    print("="*55)

    from transformers import BertTokenizer, BertModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertModel.from_pretrained("bert-base-uncased")
    model.eval().to(device)

    text_files = sorted(Path(TEXT_DIR).glob("*.textonly"))
    print(f"  Found {len(text_files)} files\n")

    skipped = []
    for f in text_files:
        out_path = Path(OUTPUT_DIR, "text", f.stem + ".npy")
        if out_path.exists():
            print(f"  [SKIP]  {f.stem}")
            continue

        try:
            text = f.read_text(encoding="utf-8").strip()
            if not text:
                skipped.append(f.stem)
                continue

            # Tokenize — max 512 BERT tokens, truncate if longer
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            ).to(device)

            with torch.no_grad():
                output = model(**tokens)

            # Remove [CLS] and [SEP] boundary tokens
            # Result: (seq_len, 768) — one vector per word
            word_vectors = output.last_hidden_state[0, 1:-1, :].cpu().numpy()

            np.save(out_path, word_vectors)
            print(f"  [OK]    {f.stem:30} => {word_vectors.shape}")

        except Exception as e:
            print(f"  [ERROR] {f.stem}: {e}")
            skipped.append(f.stem)

    print(f"\n  Text done. Skipped: {skipped}\n")


# ═══════════════════════════════════════════════════════
# STEP 2 — AUDIO (librosa MFCC)
# What it does: reads each .wav, computes acoustic
# features per time frame: pitch, energy, voice texture.
# Output shape: (n_frames, 74)
#   40 MFCCs + 20 delta MFCCs + 1 ZCR + 1 RMS + 12 chroma
# ═══════════════════════════════════════════════════════
def extract_audio_features():
    print("\n" + "="*55)
    print("  STEP 2 — AUDIO (librosa)")
    print("="*55)

    import librosa

    SAMPLE_RATE = 16000
    HOP_LENGTH  = 512
    N_MFCC      = 40

    audio_files = sorted(Path(AUDIO_DIR).glob("*.wav"))
    print(f"  Found {len(audio_files)} files\n")

    skipped = []
    for f in audio_files:
        out_path = Path(OUTPUT_DIR, "audio", f.stem + ".npy")
        if out_path.exists():
            print(f"  [SKIP]  {f.stem}")
            continue

        try:
            y, sr = librosa.load(f, sr=SAMPLE_RATE, mono=True)
            if len(y) == 0:
                skipped.append(f.stem)
                continue

            mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
            delta  = librosa.feature.delta(mfcc, order=1)[:20]
            zcr    = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
            rms    = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

            features = np.vstack([mfcc, delta, zcr, rms, chroma]).T  # (T, 74)

            np.save(out_path, features)
            print(f"  [OK]    {f.stem:30} => {features.shape}")

        except Exception as e:
            print(f"  [ERROR] {f.stem}: {e}")
            skipped.append(f.stem)

    print(f"\n  Audio done. Skipped: {skipped}\n")


# ═══════════════════════════════════════════════════════
# STEP 3 — VIDEO (OpenCV HOG face features)
# What it does: samples 1 frame every 15 frames (~2fps),
# detects the face, computes HOG descriptor on the face
# crop + bounding box + brightness stats.
#
# FIX APPLIED HERE:
#   OLD code had `padded = np.zeros(709)` — hardcoded.
#   Your HOG setup produces 908 features, not 709.
#   NEW code auto-detects the dimension from the first
#   real face detection and uses that for all frames.
#   No more crash.
#
# Output shape: (n_frames, 908)  ← auto-detected
# ═══════════════════════════════════════════════════════
def extract_video_features():
    print("\n" + "="*55)
    print("  STEP 3 — VIDEO (OpenCV HOG) — BUG FIXED")
    print("="*55)

    import cv2

    FRAME_SAMPLE_RATE  = 15
    FACE_SCALE         = 1.1
    FACE_MIN_NEIGHBORS = 5
    FACE_SIZE          = (48, 48)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    hog = cv2.HOGDescriptor(
        _winSize     = FACE_SIZE,
        _blockSize   = (16, 16),
        _blockStride = (8, 8),
        _cellSize    = (8, 8),
        _nbins       = 9
    )

    # ── Key fix: probe one frame to find real HOG size ──
    # Instead of hardcoding 709, we compute a dummy HOG
    # on a blank image to find the actual output size.
    dummy       = np.zeros(FACE_SIZE, dtype=np.uint8)
    dummy_hog   = hog.compute(dummy).flatten()
    HOG_DIM     = len(dummy_hog)           # will be 900
    FEATURE_DIM = HOG_DIM + 4 + 4         # HOG + bbox + stats = 908
    print(f"  HOG dim    : {HOG_DIM}")
    print(f"  Feature dim: {FEATURE_DIM}  (HOG + bbox + face stats)\n")

    video_files = sorted(Path(VIDEO_DIR).glob("*.mp4"))
    print(f"  Found {len(video_files)} files\n")

    skipped = []
    for f in video_files:
        out_path = Path(OUTPUT_DIR, "video", f.stem + ".npy")
        if out_path.exists():
            print(f"  [SKIP]  {f.stem}")
            continue

        try:
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                # c5xsKMxpXnc.mp4 is corrupted — moov atom missing
                print(f"  [CORRUPT] {f.name} — skipping")
                skipped.append(f.stem)
                continue

            frame_features = []
            frame_idx      = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_SAMPLE_RATE != 0:
                    frame_idx += 1
                    continue

                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=FACE_SCALE,
                    minNeighbors=FACE_MIN_NEIGHBORS,
                    minSize=(30, 30)
                )

                if len(faces) == 0:
                    # No face in frame — save zeros (same dim)
                    frame_features.append(np.zeros(FEATURE_DIM))
                    frame_idx += 1
                    continue

                # Use the largest face detected
                x, y, w, h   = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
                face_crop    = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, FACE_SIZE)

                # HOG features
                hog_feat = hog.compute(face_resized).flatten()  # (900,)

                # Normalised bounding box (4,)
                fh, fw = frame.shape[:2]
                bbox   = np.array([x/fw, y/fh, w/fw, h/fh])

                # Face appearance stats (4,)
                face_stats = np.array([
                    face_resized.mean() / 255.0,   # brightness
                    face_resized.std()  / 255.0,   # contrast
                    float(len(faces)),              # face count
                    1.0                             # detected flag
                ])

                # Combine → exactly FEATURE_DIM = 908
                combined = np.concatenate([hog_feat, bbox, face_stats])
                frame_features.append(combined)
                frame_idx += 1

            cap.release()

            if not frame_features:
                print(f"  [EMPTY] {f.name}")
                skipped.append(f.stem)
                continue

            features = np.array(frame_features)   # (n_frames, 908)
            np.save(out_path, features)
            print(f"  [OK]    {f.stem:30} => {features.shape}")

        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
            skipped.append(f.stem)

    print(f"\n  Video done. Skipped ({len(skipped)}): {skipped}\n")


# ═══════════════════════════════════════════════════════
# STEP 4 — VERIFY ALL THREE MODALITIES
# ═══════════════════════════════════════════════════════
def verify_features():
    print("\n" + "="*55)
    print("  STEP 4 — VERIFICATION")
    print("="*55)

    t_files = sorted(Path(OUTPUT_DIR, "text").glob("*.npy"))
    a_files = sorted(Path(OUTPUT_DIR, "audio").glob("*.npy"))
    v_files = sorted(Path(OUTPUT_DIR, "video").glob("*.npy"))

    print(f"\n  Extracted:")
    print(f"    Text  : {len(t_files)} files")
    print(f"    Audio : {len(a_files)} files")
    print(f"    Video : {len(v_files)} files")

    t_ids   = {f.stem for f in t_files}
    a_ids   = {f.stem for f in a_files}
    v_ids   = {f.stem for f in v_files}
    aligned = sorted(t_ids & a_ids & v_ids)

    print(f"\n  Fully aligned (all 3 modalities): {len(aligned)}")

    # Show shapes for 3 samples
    print(f"\n  Sample shapes:")
    for stem in aligned[:3]:
        t = np.load(Path(OUTPUT_DIR, "text",  stem + ".npy"))
        a = np.load(Path(OUTPUT_DIR, "audio", stem + ".npy"))
        v = np.load(Path(OUTPUT_DIR, "video", stem + ".npy"))
        print(f"\n    {stem}")
        print(f"      Text  (seq_len, 768) : {t.shape}")
        print(f"      Audio (frames,   74) : {a.shape}")
        print(f"      Video (frames,  908) : {v.shape}")

    # Save aligned ID list for next step (model training)
    import pandas as pd
    out_csv = os.path.join(OUTPUT_DIR, "aligned_ids.csv")
    pd.DataFrame(aligned, columns=["video_id"]).to_csv(out_csv, index=False)
    print(f"\n  Saved aligned IDs → {out_csv}")
    print("\n  Feature extraction COMPLETE.")
    print("  Next: sequence alignment + padding → model training")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "#"*55)
    print("  MULTIMODAL FEATURE EXTRACTION — CMU-MOSI")
    print("#"*55)

    extract_text_features()   # already done — will SKIP all 93
    extract_audio_features()  # already done — will SKIP all 93
    extract_video_features()  # FIXED — will now process all 92
    verify_features()
        