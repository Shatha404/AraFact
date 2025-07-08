import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Load Model & Tokenizer
checkpoint_path = "results/checkpoint-730"  # Change if needed

try:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    print("Model & Tokenizer Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Model: {e}")
    model, tokenizer = None, None

# Move Model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model:
    model.to(device)

# FastAPI App Setup
app = FastAPI(title="Arabic Text Classifier API")

# Mount Static Files (for images, CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 Templates Setup
templates = Jinja2Templates(directory="templates")

# Label Mapping
label_mapping = {
    0: "False",
    1: "Partly-false",
    2: "True",
    3: "Sarcasm",
    4: "Unverifiable"
}

# Homepage Route
@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# Prediction Function
def predict_label(text):
    if not model or not tokenizer:
        return "❌ Model is not loaded!"

    if not text.strip():
        return "⚠️ Please enter some text!"

    print(f"🔍 Received Text: {text}")  # Debug Log

    # Tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get Predicted Class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Debug Logs
    print(f"📊 Model Logits: {logits}")
    print(f"🎯 Predicted Class: {predicted_class}")

    # Return Prediction
    return label_mapping.get(predicted_class, "Unknown")

# Form Submission Route
@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, text: str = Form(...)):
    prediction = predict_label(text)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "input_text": text})

# Run Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
