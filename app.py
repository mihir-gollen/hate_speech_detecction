import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
import pandas as pd
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="YouTube Hate Speech Batch Analyzer", layout="centered")

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

# -----------------------------
# Helper Functions
# -----------------------------

def extract_video_id(url):
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed.query).get("v", [None])[0]
    return None

def fetch_comments(video_id, max_results=50):
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        st.error("YouTube API key not found.")
        st.stop()

    youtube = build("youtube", "v3", developerKey=api_key)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )

    response = request.execute()

    comments = []
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

def predict(text):
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

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()

    return LABEL_MAP[pred], round(confidence * 100, 2)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📺 YouTube Hate Speech Batch Analyzer")
st.write("Fetch latest 50 comments from a YouTube video and classify them.")

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Fetch & Analyze 50 Comments"):

    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL.")
    else:
        with st.spinner("Fetching comments..."):
            comments = fetch_comments(video_id)

        results = []

        with st.spinner("Analyzing comments..."):
            for comment in comments:
                label, confidence = predict(comment)
                results.append({
                    "Comment": comment,
                    "Prediction": label,
                    "Confidence (%)": confidence
                })

        df = pd.DataFrame(results)

        st.subheader("Results")
        st.dataframe(df)

        st.subheader("Prediction Distribution")
        st.bar_chart(df["Prediction"].value_counts())