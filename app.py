import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("starlinskii/arxiv/model_out")
    model = AutoModelForSequenceClassification.from_pretrained("starlinskii/arxiv/model_out")
    model.eval()
    return tokenizer, model


def build_text(title, abstract):
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if abstract:
        return f"{title} {abstract}"
    return title

def predict_top95(title, abstract, tokenizer, model):
    text = build_text(title, abstract)
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()
    pairs = []
    for i, p in enumerate(probs):
        label = model.config.id2label[i]
        pairs.append((label, p))
    pairs.sort(key=lambda x: x[1], reverse=True)
    result = []
    cumulative = 0.0
    for label, prob in pairs:
        result.append((label, prob))
        cumulative += prob
        if cumulative >= 0.95:
            break
    return result


st.title("Article Topic Classifier")
tokenizer, model = load_model()
title = st.text_input("Title")
abstract = st.text_area("Abstract")
if st.button("Classify"):
    if not title.strip():
        st.error("Title is required")
    else:
        preds = predict_top95(title, abstract, tokenizer, model)
        st.subheader("Top topics")
        for label, prob in preds:
            st.write(f"{label} — {prob:.4f}")
