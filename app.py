import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(page_title="Hate Speech Detection", layout="centered")

MODEL_PATH = "saved_model/model"
TOKENIZER_PATH = "saved_model/tokenizer"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

LABEL_MAP = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

st.title("🛡️ Online Hate Speech Detection")
st.write("Enter text below to classify it using a transformer-based NLP model.")

text = st.text_area("Input Text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE)
            )

        pred = torch.argmax(outputs.logits, dim=1).item()
        st.success(f"Prediction: **{LABEL_MAP[pred]}**")
