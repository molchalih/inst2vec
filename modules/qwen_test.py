import sys
import os
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder

MODEL_PATH = "./models/Qwen3-VL-Embedding-8B"
VIDEOS_DIR = "./data/source/videos"

videos = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")]
video_file = random.choice(videos)
video_path = os.path.abspath(os.path.join(VIDEOS_DIR, video_file))

print(f"Selected video: {video_file}")
print(f"Loading model from: {MODEL_PATH}")

model = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH)
print(f"Device: {model.model.device}\n")

inputs = [
    {
        "video": video_path,
        "instruction": "Represent the video content for retrieval.",
    },
    {
        "text": "A person dancing to music in a short video.",
    },
    {
        "text": "A cooking tutorial showing how to prepare a meal.",
    },
    {
        "text": "A cat sleeping on a windowsill.",
    },
]

print(f"Embedding 1 video + 3 text queries...\n")

embeddings = model.process(inputs)

print(f"Embedding shape: {embeddings.shape}")
print(f"\nSimilarity matrix:")
similarity = embeddings @ embeddings.T
print(similarity)

labels = [
    f"[video] {video_file}",
    "[text] A person dancing to music...",
    "[text] A cooking tutorial...",
    "[text] A cat sleeping...",
]

print(f"\nVideo vs text similarities:")
for j in range(1, len(labels)):
    print(f"  {labels[0]} vs {labels[j]}: {similarity[0][j].item():.4f}")
