import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = r"D:\multimodel\dataset"
ALIGN_DIR  = os.path.join(BASE_DIR, "aligned")
OUT_DIR    = os.path.join(BASE_DIR, "scores")
os.makedirs(OUT_DIR, exist_ok=True)

# Feature dims (from your extraction)
TEXT_DIM   = 768
AUDIO_DIM  = 74
VIDEO_DIM  = 908

# Model dims
PROJ_DIM   = 128    # all modalities projected to this
HIDDEN_DIM = 256    # after fusion
NUM_HEADS  = 4      # attention heads
DROPOUT    = 0.3

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ═══════════════════════════════════════════════════════
# STEP 1 — DATASET
# Loads aligned .npy files for all 3 modalities
# ═══════════════════════════════════════════════════════
class MOSIDataset(Dataset):
    def __init__(self, align_dir):
        self.text  = np.load(os.path.join(align_dir, "text_data.npy"))
        self.audio = np.load(os.path.join(align_dir, "audio_data.npy"))
        self.video = np.load(os.path.join(align_dir, "video_data.npy"))

        self.text_mask  = np.load(os.path.join(align_dir, "text_mask.npy"))
        self.audio_mask = np.load(os.path.join(align_dir, "audio_mask.npy"))
        self.video_mask = np.load(os.path.join(align_dir, "video_mask.npy"))

        ids_path = os.path.join(align_dir, "video_ids.csv")
        self.video_ids = pd.read_csv(ids_path)["video_id"].tolist()

        print(f"  Loaded {len(self.video_ids)} samples")
        print(f"  Text  : {self.text.shape}")
        print(f"  Audio : {self.audio.shape}")
        print(f"  Video : {self.video.shape}")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        return {
            "text"       : torch.tensor(self.text[idx],        dtype=torch.float32),
            "audio"      : torch.tensor(self.audio[idx],       dtype=torch.float32),
            "video"      : torch.tensor(self.video[idx],       dtype=torch.float32),
            "text_mask"  : torch.tensor(self.text_mask[idx],   dtype=torch.float32),
            "audio_mask" : torch.tensor(self.audio_mask[idx],  dtype=torch.float32),
            "video_mask" : torch.tensor(self.video_mask[idx],  dtype=torch.float32),
            "video_id"   : self.video_ids[idx],
        }


# ═══════════════════════════════════════════════════════
# STEP 2 — CNN ENCODER
# Compresses raw features to common dim (PROJ_DIM=128)
# Input : (B, T, D_raw)
# Output: (B, T, 128)
# ═══════════════════════════════════════════════════════
class CNNEncoder(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
        self.cnn  = nn.Sequential(
            # Conv1d expects (B, C, T) — we treat features as channels
            nn.Conv1d(proj_dim, proj_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(proj_dim, proj_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(proj_dim)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        # x: (B, T, D_raw)
        x = self.proj(x)                      # (B, T, proj_dim)
        x = x.transpose(1, 2)                 # (B, proj_dim, T) for Conv1d
        x = self.cnn(x)                       # (B, proj_dim, T)
        x = x.transpose(1, 2)                 # (B, T, proj_dim)
        x = self.norm(x)
        x = self.drop(x)
        return x


# ═══════════════════════════════════════════════════════
# STEP 3 — CROSS MODAL ATTENTION
# One modality (query) attends to another (key/value)
# Ft' = Attention(Ft, Fv)  — text attends to video
# Fa' = Attention(Fa, Ft)  — audio attends to text
# Fv' = Attention(Fv, Fa)  — video attends to audio
# ═══════════════════════════════════════════════════════
class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # Q comes from query modality, K/V from key modality
        self.attn = nn.MultiheadAttention(
            embed_dim    = dim,
            num_heads    = num_heads,
            dropout      = DROPOUT,
            batch_first  = True      # (B, T, D) format
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, query, key_value, query_mask=None, kv_mask=None):
        """
        query      : (B, Tq, D) — the modality asking
        key_value  : (B, Tk, D) — the modality being asked
        query_mask : (B, Tq)    — 1=real, 0=padding
        kv_mask    : (B, Tk)    — 1=real, 0=padding

        Returns enhanced query: (B, Tq, D)
        """
        # Convert masks: MultiheadAttention expects True=IGNORE
        kv_key_mask = None
        if kv_mask is not None:
            kv_key_mask = (kv_mask == 0)   # (B, Tk) True where padded

        attended, attn_weights = self.attn(
            query   = query,
            key     = key_value,
            value   = key_value,
            key_padding_mask = kv_key_mask
        )

        # Residual connection + norm
        out = self.norm(query + self.drop(attended))
        return out, attn_weights


# ═══════════════════════════════════════════════════════
# STEP 4 — RELIABILITY ESTIMATOR
# Evaluates how trustworthy each modality is per sample
# Output: scalar reliability score per modality (0–1)
# ═══════════════════════════════════════════════════════
class ReliabilityEstimator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Pool across time, then score
        self.pool = nn.AdaptiveAvgPool1d(1)     # (B, D, T) → (B, D, 1)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()                        # output 0–1
        )

    def forward(self, x, mask=None):
        """
        x    : (B, T, D)
        mask : (B, T) — 1=real, 0=padding
        Returns: (B, 1) reliability score per sample
        """
        if mask is not None:
            # Zero out padded positions before pooling
            m = mask.unsqueeze(-1)              # (B, T, 1)
            x = x * m

        # Pool across time dimension
        x_t = x.transpose(1, 2)               # (B, D, T)
        pooled = self.pool(x_t).squeeze(-1)    # (B, D)

        score = self.mlp(pooled)               # (B, 1)
        return score


# ═══════════════════════════════════════════════════════
# STEP 5 — FULL MODEL
# Ties everything together:
#   CNN encode → cross-modal attention →
#   reliability estimation → dynamic weighted fusion →
#   classifier → sentiment score
# ═══════════════════════════════════════════════════════
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN encoders — project each modality to PROJ_DIM
        self.text_enc  = CNNEncoder(TEXT_DIM,  PROJ_DIM)
        self.audio_enc = CNNEncoder(AUDIO_DIM, PROJ_DIM)
        self.video_enc = CNNEncoder(VIDEO_DIM, PROJ_DIM)

        # Cross-modal attention (3 pairs)
        self.text_attn  = CrossModalAttention(PROJ_DIM, NUM_HEADS)  # text ← video
        self.audio_attn = CrossModalAttention(PROJ_DIM, NUM_HEADS)  # audio ← text
        self.video_attn = CrossModalAttention(PROJ_DIM, NUM_HEADS)  # video ← audio

        # Reliability estimators (one per modality)
        self.text_rel  = ReliabilityEstimator(PROJ_DIM)
        self.audio_rel = ReliabilityEstimator(PROJ_DIM)
        self.video_rel = ReliabilityEstimator(PROJ_DIM)

        # Classifier head
        # Input: fused vector (PROJ_DIM) → 3 classes
        self.classifier = nn.Sequential(
            nn.Linear(PROJ_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 3)   # Positive / Neutral / Negative
        )

    def forward(self, text, audio, video,
                text_mask=None, audio_mask=None, video_mask=None):
        """
        text   : (B, 50,  768)
        audio  : (B, 400, 74)
        video  : (B, 300, 908)

        Returns:
          logits      : (B, 3)   raw class scores
          probs       : (B, 3)   softmax probabilities
          weights     : (B, 3)   [wt, wa, wv] dynamic weights
          reliability : (B, 3)   [Rt, Ra, Rv] reliability scores
        """

        # ── STEP A: CNN encode all modalities → (B, T, 128) ──
        Ft = self.text_enc(text)    # (B, 50,  128)
        Fa = self.audio_enc(audio)  # (B, 400, 128)
        Fv = self.video_enc(video)  # (B, 300, 128)

        # ── STEP B: Cross-modal attention ──
        # Text attends to video  → Ft'
        Ft_prime, tw = self.text_attn(
            Ft, Fv,
            query_mask = text_mask,
            kv_mask    = video_mask
        )

        # Audio attends to text  → Fa'
        Fa_prime, aw = self.audio_attn(
            Fa, Ft,
            query_mask = audio_mask,
            kv_mask    = text_mask
        )

        # Video attends to audio → Fv'
        Fv_prime, vw = self.video_attn(
            Fv, Fa,
            query_mask = video_mask,
            kv_mask    = audio_mask
        )

        # ── STEP C: Reliability estimation ──
        # Each modality gets a trustworthiness score 0–1
        Rt = self.text_rel(Ft_prime,  text_mask)    # (B, 1)
        Ra = self.audio_rel(Fa_prime, audio_mask)   # (B, 1)
        Rv = self.video_rel(Fv_prime, video_mask)   # (B, 1)

        reliability = torch.cat([Rt, Ra, Rv], dim=1)  # (B, 3)

        # ── STEP D: Dynamic weights via softmax ──
        # [wt, wa, wv] always sum to 1
        weights = F.softmax(reliability, dim=1)        # (B, 3)
        wt = weights[:, 0:1]   # (B, 1)
        wa = weights[:, 1:2]   # (B, 1)
        wv = weights[:, 2:3]   # (B, 1)

        # ── STEP E: Pool each modality across time ──
        # Then apply dynamic weights
        def masked_pool(feat, mask):
            if mask is not None:
                feat = feat * mask.unsqueeze(-1)
                count = mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8
                return feat.sum(dim=1) / count.squeeze(-1)
            return feat.mean(dim=1)

        ft_vec = masked_pool(Ft_prime, text_mask)    # (B, 128)
        fa_vec = masked_pool(Fa_prime, audio_mask)   # (B, 128)
        fv_vec = masked_pool(Fv_prime, video_mask)   # (B, 128)

        # ── STEP F: Weighted fusion ──
        # F = wt·Ft' + wa·Fa' + wv·Fv'
        fused = wt * ft_vec + wa * fa_vec + wv * fv_vec   # (B, 128)

        # ── STEP G: Classify ──
        logits = self.classifier(fused)                    # (B, 3)
        probs  = F.softmax(logits, dim=1)                  # (B, 3)

        return logits, probs, weights, reliability


# ═══════════════════════════════════════════════════════
# STEP 6 — SCORE GENERATOR
# Runs the model on all 92 videos and saves scores
# even without labels — useful to verify the pipeline
# ═══════════════════════════════════════════════════════
def generate_scores():
    print("\n" + "="*55)
    print("  GENERATING SENTIMENT SCORES")
    print("="*55)

    # Load dataset
    dataset    = MOSIDataset(ALIGN_DIR)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Build model
    model = MultimodalSentimentModel().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {total_params:,}")

    model.eval()
    results = []

    CLASS_NAMES = ["Negative", "Neutral", "Positive"]

    with torch.no_grad():
        for batch in dataloader:
            text   = batch["text"].to(DEVICE)
            audio  = batch["audio"].to(DEVICE)
            video  = batch["video"].to(DEVICE)
            tmask  = batch["text_mask"].to(DEVICE)
            amask  = batch["audio_mask"].to(DEVICE)
            vmask  = batch["video_mask"].to(DEVICE)
            vids   = batch["video_id"]

            logits, probs, weights, reliability = model(
                text, audio, video, tmask, amask, vmask
            )

            # Convert to numpy
            probs_np  = probs.cpu().numpy()
            weights_np = weights.cpu().numpy()
            rel_np    = reliability.cpu().numpy()

            for i, vid in enumerate(vids):
                pred_class = CLASS_NAMES[probs_np[i].argmax()]
                results.append({
                    "video_id"          : vid,
                    "pred_class"        : pred_class,
                    "prob_negative"     : round(float(probs_np[i][0]),  4),
                    "prob_neutral"      : round(float(probs_np[i][1]),  4),
                    "prob_positive"     : round(float(probs_np[i][2]),  4),
                    "confidence"        : round(float(probs_np[i].max()), 4),
                    "weight_text"       : round(float(weights_np[i][0]), 4),
                    "weight_audio"      : round(float(weights_np[i][1]), 4),
                    "weight_video"      : round(float(weights_np[i][2]), 4),
                    "reliability_text"  : round(float(rel_np[i][0]),    4),
                    "reliability_audio" : round(float(rel_np[i][1]),    4),
                    "reliability_video" : round(float(rel_np[i][2]),    4),
                })

    # Save results
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "sentiment_scores.csv")
    df.to_csv(out_csv, index=False)

    return df


# ═══════════════════════════════════════════════════════
# STEP 7 — PRINT SUMMARY
# ═══════════════════════════════════════════════════════
def print_summary(df):
    print("\n" + "="*55)
    print("  SCORE SUMMARY")
    print("="*55)

    print(f"\n  Total videos scored: {len(df)}")

    # Class distribution
    counts = df["pred_class"].value_counts()
    print(f"\n  Prediction distribution:")
    for cls, cnt in counts.items():
        bar = "█" * cnt
        print(f"    {cls:10} {cnt:3}  {bar}")

    # Average modality weights
    print(f"\n  Average dynamic weights across all videos:")
    print(f"    Text  : {df['weight_text'].mean():.3f}")
    print(f"    Audio : {df['weight_audio'].mean():.3f}")
    print(f"    Video : {df['weight_video'].mean():.3f}")

    print(f"\n  Average reliability scores:")
    print(f"    Text  : {df['reliability_text'].mean():.3f}")
    print(f"    Audio : {df['reliability_audio'].mean():.3f}")
    print(f"    Video : {df['reliability_video'].mean():.3f}")

    # Sample predictions
    print(f"\n  Sample predictions (first 10):")
    print(f"  {'Video ID':30} {'Pred':10} {'Conf':6} {'Wt':6} {'Wa':6} {'Wv':6}")
    print(f"  {'-'*66}")
    for _, row in df.head(10).iterrows():
        print(f"  {row['video_id']:30} {row['pred_class']:10} "
              f"{row['confidence']:.3f}  "
              f"{row['weight_text']:.3f}  "
              f"{row['weight_audio']:.3f}  "
              f"{row['weight_video']:.3f}")

    out_csv = os.path.join(OUT_DIR, "sentiment_scores.csv")
    print(f"\n  Full results saved to: {out_csv}")
    print("\n  NOTE: These are untrained model scores (random weights).")
    print("  To get meaningful scores you need to train the model.")
    print("  Next step: load MOSI labels + train the model.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "#"*55)
    print("  MULTIMODAL SENTIMENT SCORING PIPELINE")
    print("#"*55)

    df = generate_scores()
    print_summary(df)