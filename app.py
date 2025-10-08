import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

def load_ocr_pipeline():
    try:
        pipe = pipeline(
            "image-to-text", 
            model="microsoft/trocr-base-printed",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return pipe
    except:
        pipe = pipeline(
            "image-to-text",
            model="microsoft/trocr-base-stage1",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return pipe