"""Streamlit demo for single-cascade misinformation prediction."""

import io
import os
import sys

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.checkpoints import load_checkpoint
from pipeline.config import load_config
from pipeline.inference import cascade_from_arrays, predict_cascade


st.set_page_config(page_title="Misinformation Drift Detector", page_icon="")
st.title("Misinformation Drift Detector")
st.caption("Research prototype: graph-sequence classification with propagation drift analysis")

config_path = st.sidebar.text_input("Config", "configs/default.yaml")
checkpoint_path = st.sidebar.text_input("Checkpoint", "checkpoints/hybrid_gossipcop_seed1.pt")
uploaded = st.file_uploader("Upload a cascade (.npz)", type=["npz"])

if not os.path.isfile(checkpoint_path):
    st.info("Train a checkpoint first, then enter its path in the sidebar.")
    st.code(
        "python3 multiseed.py --subset gossipcop --models Hybrid --seeds 1 "
        "--out results/_canonical_hybrid_seed1.json --checkpoint-dir checkpoints"
    )
    st.stop()

if uploaded is None:
    st.info("Generate an example with scripts/create_sample_cascade.py or upload an UPFD-format .npz file.")
    st.stop()

try:
    with np.load(io.BytesIO(uploaded.getvalue()), allow_pickle=False) as arrays:
        graph = cascade_from_arrays({key: arrays[key] for key in arrays.files})
    config = load_config(config_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, metadata = load_checkpoint(
        checkpoint_path, device, model_config=config.get("model")
    )
    if graph.x.size(1) != int(metadata.get("input_dim", graph.x.size(1))):
        raise ValueError("Uploaded feature dimension does not match the checkpoint")
    result = predict_cascade(model, graph, device)
except Exception as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns(2)
left.metric("Prediction", result["label"].upper())
right.metric("Fake probability", f"{result['probability_fake']:.1%}")
st.metric("Confidence", f"{result['confidence']:.1%}")

st.subheader("Cascade summary")
st.json({key: result[key] for key in (
    "cascade_size", "feature_dim", "max_depth", "mean_cosine_drift",
    "mean_l2_drift", "mean_content_drift",
)})
st.subheader("Mean content drift by hop")
st.line_chart(result["content_drift_by_hop"])
st.caption("This prototype is for research and should not be used as a real-world fact-checking system.")
