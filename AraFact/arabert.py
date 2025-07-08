import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import (
    Trainer,
    TrainingArguments,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer, AutoModelForSequenceClassification,
)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import evaluate

# Load dataset
df = pd.read_csv('cleaned_claim (2).csv')

print(df.head())



# Print category counts
category_counts = df['normalized_label'].value_counts()
for category, count in category_counts.items():
    print(f"Category: {category}, Rows: {count}")

df["normalized_label"] = df["normalized_label"].str.strip().str.lower()  # ✅ Remove spaces & lowercase
label_mapping = {
    "false": 0,
    "partly-false": 1,
    "true": 2,
    "sarcasm": 3,
    "unverifiable": 4,
}
df["Mapped_Labels"] = df["normalized_label"].map(label_mapping)


print(df.head())


# Load tokenizer
model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["cleaned_claim"], 
        padding="max_length",  # ✅ Ensures uniform length
        truncation=True,       # ✅ Prevents excessive token length
        max_length=128,        # ✅ Adjust as needed
    )

from datasets import Dataset

# Ensure correct keys in dataset dictionary
train_dataset = Dataset.from_pandas(df[["cleaned_claim", "Mapped_Labels"]])
test_dataset = Dataset.from_pandas(df[["cleaned_claim", "Mapped_Labels"]])

# Rename label column to match Hugging Face's convention
train_dataset = train_dataset.rename_columns({"Mapped_Labels": "label"})
test_dataset = test_dataset.rename_columns({"Mapped_Labels": "label"})


# Tokenize dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load pre-trained arabert model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
print("AraBert Model and tokenizer loaded successfully!")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load evaluation metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",
    fp16=False,  
    gradient_accumulation_steps=2,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)


# Move model to GPU
model.to(device)

# Define the trainer parameters
trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train using GPU
trainer.train()

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure model is in evaluation mode
model.eval()

# Get predictions using Trainer
predictions = trainer.predict(test_dataset)

# Extract logits & true labels
logits = predictions.predictions  # Raw model outputs
y_pred = np.argmax(logits, axis=-1)  # Convert logits to class labels
y_true = predictions.label_ids  # Actual labels


# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=label_mapping.keys()))

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()