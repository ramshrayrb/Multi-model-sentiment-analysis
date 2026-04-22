import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = r"D:\multimodel\dataset"
FEAT_DIR   = os.path.join(BASE_DIR, "features")

TEXT_DIR   = os.path.join(FEAT_DIR, "text")
AUDIO_DIR  = os.path.join(FEAT_DIR, "audio")
VIDEO_DIR  = os.path.join(FEAT_DIR, "video")
ALIGN_DIR  = os.path.join(BASE_DIR, "aligned")
os.makedirs(ALIGN_DIR, exist_ok=True)

# Fixed lengths after padding/truncation
MAX_TEXT   = 50     # word tokens
MAX_AUDIO  = 400    # audio frames
MAX_VIDEO  = 300    # video frames

# Feature dims (from your extraction)
TEXT_DIM   = 768
AUDIO_DIM  = 74
VIDEO_DIM  = 908


# ═══════════════════════════════════════════════════════
# STEP 1 — INSPECT RAW SHAPES BEFORE PADDING
# So you can confirm max lengths make sense
# ═══════════════════════════════════════════════════════
def inspect_shapes():
    print("\n" + "="*55)
    print("  STEP 1 — INSPECTING RAW FEATURE SHAPES")
    print("="*55)

    for modality, folder in [("TEXT", TEXT_DIR),
                              ("AUDIO", AUDIO_DIR),
                              ("VIDEO", VIDEO_DIR)]:
        files = sorted(Path(folder).glob("*.npy"))
        if not files:
            print(f"\n  {modality}: no files found in {folder}")
            continue

        lengths = []
        dims    = []
        for f in files:
            arr = np.load(f)
            lengths.append(arr.shape[0])
            dims.append(arr.shape[1] if arr.ndim > 1 else 1)

        print(f"\n  {modality} ({len(files)} files)")
        print(f"    Seq length — min:{min(lengths):5}  "
              f"max:{max(lengths):5}  "
              f"mean:{int(np.mean(lengths)):5}  "
              f"median:{int(np.median(lengths)):5}")
        print(f"    Feature dim — {dims[0]}")
        print(f"    Suggested MAX_LEN — {int(np.percentile(lengths, 90))}  "
              f"(covers 90% of samples)")


# ═══════════════════════════════════════════════════════
# STEP 2 — PAD / TRUNCATE ONE ARRAY
# ═══════════════════════════════════════════════════════
def pad_or_truncate(arr, max_len, feat_dim):
    """
    Input : arr of shape (T, D)
    Output: array of shape (max_len, D)
      - if T > max_len → truncate first max_len rows
      - if T < max_len → pad with zeros at the end
    Also returns a mask: 1 where real data, 0 where padding
    """
    # Safety: handle 1-D arrays
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    T = arr.shape[0]
    mask = np.zeros(max_len, dtype=np.float32)

    if T >= max_len:
        out = arr[:max_len, :feat_dim]
        mask[:] = 1.0                          # all real
    else:
        pad = np.zeros((max_len - T, feat_dim), dtype=np.float32)
        out = np.vstack([arr[:, :feat_dim], pad])
        mask[:T] = 1.0                         # only first T are real

    return out.astype(np.float32), mask


# ═══════════════════════════════════════════════════════
# STEP 3 — LOAD ALL FILES FOR ONE MODALITY
# ═══════════════════════════════════════════════════════
def load_modality(folder, video_ids, max_len, feat_dim, label):
    print(f"\n  Loading {label}...")
    data  = []
    masks = []

    for vid in video_ids:
        path = Path(folder, vid + ".npy")

        if not path.exists():
            # Missing file → all zeros + zero mask
            print(f"    [MISSING] {vid}")
            data.append(np.zeros((max_len, feat_dim), dtype=np.float32))
            masks.append(np.zeros(max_len, dtype=np.float32))
            continue

        arr = np.load(path).astype(np.float32)

        # Handle edge case: empty array
        if arr.size == 0:
            data.append(np.zeros((max_len, feat_dim), dtype=np.float32))
            masks.append(np.zeros(max_len, dtype=np.float32))
            continue

        padded, mask = pad_or_truncate(arr, max_len, feat_dim)
        data.append(padded)
        masks.append(mask)

    data_arr  = np.array(data)    # (N, max_len, feat_dim)
    masks_arr = np.array(masks)   # (N, max_len)
    print(f"    Shape: {data_arr.shape}  Mask: {masks_arr.shape}")
    return data_arr, masks_arr


# ═══════════════════════════════════════════════════════
# STEP 4 — NORMALIZE FEATURES (zero mean, unit variance)
# Done per feature dimension across all samples
# ═══════════════════════════════════════════════════════
def normalize(data, mask=None):
    """
    data : (N, T, D)
    Compute mean and std only over real (non-padded) positions.
    """
    N, T, D = data.shape

    if mask is not None:
        # Expand mask to (N, T, D) for broadcasting
        m = mask[:, :, np.newaxis]             # (N, T, 1)
        real_sum   = (data * m).sum(axis=(0, 1))
        real_count = m.sum(axis=(0, 1)) + 1e-8
        mean = real_sum / real_count           # (D,)

        sq_diff = ((data - mean) * m) ** 2
        std = np.sqrt(sq_diff.sum(axis=(0, 1)) / real_count + 1e-8)
    else:
        mean = data.mean(axis=(0, 1))
        std  = data.std(axis=(0, 1)) + 1e-8

    normed = (data - mean) / std
    return normed.astype(np.float32), mean, std


# ═══════════════════════════════════════════════════════
# STEP 5 — FULL ALIGNMENT PIPELINE
# ═══════════════════════════════════════════════════════
def run_alignment():
    print("\n" + "="*55)
    print("  STEP 3–5 — SEQUENCE ALIGNMENT & NORMALIZATION")
    print("="*55)

    # ── Find aligned video IDs (present in all 3 modalities) ──
    text_ids  = {f.stem for f in Path(TEXT_DIR).glob("*.npy")}
    audio_ids = {f.stem for f in Path(AUDIO_DIR).glob("*.npy")}
    video_ids = {f.stem for f in Path(VIDEO_DIR).glob("*.npy")}
    aligned   = sorted(text_ids & audio_ids & video_ids)

    print(f"\n  Aligned video IDs: {len(aligned)}")
    if not aligned:
        print("  ERROR: No aligned IDs found. Check your feature folders.")
        return

    # ── Load & pad each modality ──
    text_data,  text_mask  = load_modality(
        TEXT_DIR,  aligned, MAX_TEXT,  TEXT_DIM,  "TEXT  (50 tokens, 768-dim)")
    audio_data, audio_mask = load_modality(
        AUDIO_DIR, aligned, MAX_AUDIO, AUDIO_DIM, "AUDIO (400 frames, 74-dim)")
    video_data, video_mask = load_modality(
        VIDEO_DIR, aligned, MAX_VIDEO, VIDEO_DIM, "VIDEO (300 frames, 908-dim)")

    # ── Normalize each modality ──
    print("\n  Normalizing features (zero mean, unit variance)...")
    text_norm,  text_mean,  text_std  = normalize(text_data,  text_mask)
    audio_norm, audio_mean, audio_std = normalize(audio_data, audio_mask)
    video_norm, video_mean, video_std = normalize(video_data, video_mask)

    # ── Save aligned tensors ──
    print("\n  Saving aligned tensors...")

    np.save(os.path.join(ALIGN_DIR, "text_data.npy"),   text_norm)
    np.save(os.path.join(ALIGN_DIR, "audio_data.npy"),  audio_norm)
    np.save(os.path.join(ALIGN_DIR, "video_data.npy"),  video_norm)

    np.save(os.path.join(ALIGN_DIR, "text_mask.npy"),   text_mask)
    np.save(os.path.join(ALIGN_DIR, "audio_mask.npy"),  audio_mask)
    np.save(os.path.join(ALIGN_DIR, "video_mask.npy"),  video_mask)

    # Save normalization stats (needed at inference time)
    np.save(os.path.join(ALIGN_DIR, "text_mean.npy"),   text_mean)
    np.save(os.path.join(ALIGN_DIR, "text_std.npy"),    text_std)
    np.save(os.path.join(ALIGN_DIR, "audio_mean.npy"),  audio_mean)
    np.save(os.path.join(ALIGN_DIR, "audio_std.npy"),   audio_std)
    np.save(os.path.join(ALIGN_DIR, "video_mean.npy"),  video_mean)
    np.save(os.path.join(ALIGN_DIR, "video_std.npy"),   video_std)

    # Save video ID list (order matters!)
    pd.DataFrame(aligned, columns=["video_id"]).to_csv(
        os.path.join(ALIGN_DIR, "video_ids.csv"), index=False)

    return aligned, text_norm, audio_norm, video_norm


# ═══════════════════════════════════════════════════════
# STEP 6 — VERIFY SAVED FILES
# ═══════════════════════════════════════════════════════
def verify():
    print("\n" + "="*55)
    print("  STEP 6 — VERIFICATION")
    print("="*55)

    files = {
        "text_data.npy"  : (None, MAX_TEXT,  TEXT_DIM),
        "audio_data.npy" : (None, MAX_AUDIO, AUDIO_DIM),
        "video_data.npy" : (None, MAX_VIDEO, VIDEO_DIM),
        "text_mask.npy"  : (None, MAX_TEXT,  None),
        "audio_mask.npy" : (None, MAX_AUDIO, None),
        "video_mask.npy" : (None, MAX_VIDEO, None),
    }

    all_ok = True
    for fname, (_, exp_t, exp_d) in files.items():
        path = os.path.join(ALIGN_DIR, fname)
        if not os.path.exists(path):
            print(f"  [MISSING] {fname}")
            all_ok = False
            continue
        arr = np.load(path)
        ok = True
        if exp_t and arr.shape[-1 if 'mask' in fname else 1] != exp_t:
            ok = False
        if exp_d and arr.ndim > 1 and arr.shape[2] != exp_d:
            ok = False
        status = "[OK] " if ok else "[SHAPE MISMATCH]"
        print(f"  {status} {fname:25} shape: {arr.shape}")
        if not ok:
            all_ok = False

    # Sample one video — show real vs padded token counts
    ids_path = os.path.join(ALIGN_DIR, "video_ids.csv")
    if os.path.exists(ids_path):
        ids  = pd.read_csv(ids_path)["video_id"].tolist()
        tmask = np.load(os.path.join(ALIGN_DIR, "text_mask.npy"))
        amask = np.load(os.path.join(ALIGN_DIR, "audio_mask.npy"))
        vmask = np.load(os.path.join(ALIGN_DIR, "video_mask.npy"))

        print(f"\n  Sample — real (non-padded) lengths for first 5 videos:")
        print(f"  {'Video ID':30} {'Text':>6} {'Audio':>6} {'Video':>6}")
        print(f"  {'-'*52}")
        for i, vid in enumerate(ids[:5]):
            t = int(tmask[i].sum())
            a = int(amask[i].sum())
            v = int(vmask[i].sum())
            print(f"  {vid:30} {t:>6} {a:>6} {v:>6}")

    if all_ok:
        print("\n  Alignment complete.")
        print("  Next step: load labels + build LSTM model")
    else:
        print("\n  Some files have issues — check above.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "#"*55)
    print("  SEQUENCE ALIGNMENT PIPELINE")
    print("#"*55)

    # Step 1 — inspect raw shapes first
    inspect_shapes()

    # Steps 3–5 — pad, normalize, save
    result = run_alignment()

    # Step 6 — verify output
    verify()