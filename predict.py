import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "saved_model/model"
TOKENIZER_PATH = "saved_model/tokenizer"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model from local files
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

LABEL_MAP = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

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

    pred = torch.argmax(outputs.logits, dim=1).item()
    return LABEL_MAP[pred]

# Test
if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or 'exit'): ")
        if text.lower() == "exit":
            break
        print("Prediction:", predict(text))
