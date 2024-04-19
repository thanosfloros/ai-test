from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import datetime

print(f"Time: {datetime.datetime.fromtimestamp(time.time())}")
# Load dataset
dataset = load_dataset("imdb")
print("Loaded dataset: ", dataset.values)

df = pd.DataFrame(dataset["train"])
print("Converted dataset to pandas DataFrame: ",df.to_json)
print("Split dataset into training and evaluation sets...")
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
print("len(train_df)=", len(train_df))
print("len(eval_df)=", len(eval_df))
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

print(f"Time: {datetime.datetime.fromtimestamp(time.time())}")
# Tokenize function

def tokenize_function(examples):
    # Tokenize text using the tokenizer
    tokenized_inputs = tokenizer(examples["text"].tolist(), padding="max_length", truncation=True, return_tensors="pt")
    print("Tokenized inputs:", tokenized_inputs)
    # Prepare inputs as a dictionary with required keys
    inputs = {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        # Optionally include token_type_ids if needed
        # "token_type_ids": tokenized_inputs["token_type_ids"]
    }
    print("Inputs:", inputs)
    return inputs

print(f"Time: {datetime.datetime.fromtimestamp(time.time())}")

# Tokenize datasets
train_dataset = tokenize_function(train_df) 
eval_dataset = tokenize_function(eval_df) 
print("Tokenized inputs for train_dataset:", train_dataset.keys())
print("Tokenized inputs for eval_dataset:", eval_dataset.keys())

# Print the first few samples of train_dataset
print("Train Dataset Sample:")
for i in range(min(5, len(train_dataset["input_ids"]))):
    sample = {key: tensor[i] for key, tensor in train_dataset.items()}
    print(sample)

# Print the first few samples of eval_dataset
print("Eval Dataset Sample:")
for i in range(min(5, len(eval_dataset["input_ids"]))):
    sample = {key: tensor[i] for key, tensor in eval_dataset.items()}
    print(sample)

print(f"Time: {datetime.datetime.fromtimestamp(time.time())}")
# Load model
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=len(dataset["train"].features["label"].names))

print("Model Identifier:", model.__class__.__name__)
print("Number of Parameters:", sum(p.numel() for p in model.parameters()))

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}
print(f"Time: {datetime.datetime.fromtimestamp(time.time())}")
# Training arguments
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",  # Evaluate each epoch
    per_device_train_batch_size=8,  # Batch size per GPU
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # Number of epochs
    logging_dir="logs",
    logging_strategy="epoch",  # Log after each epoch
)

print(f"Time: {datetime.datetime.fromtimestamp(time.time())}")
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
print(f"Time: {datetime.datetime.fromtimestamp(time.time())}")
# Train model
trainer.train()
print(f"Time: {datetime.datetime.fromtimestamp(time.time())}")
# Evaluate final model
results = trainer.evaluate(eval_dataset)
print(results)
