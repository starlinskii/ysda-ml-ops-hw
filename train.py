import os
import random
from collections import Counter
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert/distilbert-base-cased"
DATA_PATH = "/kaggle/input/datasets/organizations/Cornell-University/arxiv/arxiv-metadata-oai-snapshot.json"
FILTERED_PATH = "filtered_arxiv_8_domains.jsonl"
OUTPUT_DIR = "model_out_8_labels_full_dataset"
MAX_LENGTH = 128
SEED = 42

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_categories(categories_str: str) -> List[str]:
    if not isinstance(categories_str, str):
        return []
    return [x.strip() for x in categories_str.split() if x.strip()]

def map_category_to_domain(cat: str) -> Optional[str]:
    cat = cat.strip()

    if cat.startswith("cs."):
        return "computer_science"
    if cat.startswith("math."):
        return "mathematics"
    if cat.startswith("stat."):
        return "statistics"
    if cat.startswith("q-bio."):
        return "quantitative_biology"
    if cat.startswith("q-fin."):
        return "quantitative_finance"
    if cat.startswith("econ."):
        return "economics"
    if cat.startswith("eess."):
        return "electrical_engineering_and_systems_science"

    physics_prefixes = (
        "physics.",
        "astro-ph",
        "cond-mat",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math-ph",
        "nlin",
        "nucl-ex",
        "nucl-th",
        "quant-ph",
    )
    if cat.startswith(physics_prefixes):
        return "physics"

    return None

def choose_domain(categories_str: str) -> Optional[str]:
    for cat in parse_categories(categories_str):
        domain = map_category_to_domain(cat)
        if domain is not None:
            return domain
    return None

def tokenize_function(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
dataset = load_dataset("json", data_files=FILTERED_PATH, split="train")

label_names = sorted(set(dataset["label_name"]))
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}

def encode_label(example):
    example["label"] = label2id[example["label_name"]]
    return example

dataset = dataset.map(encode_label)
dataset = dataset.train_test_split(
    test_size=0.2,
    seed=SEED,
)

train_ds = dataset["train"]
val_ds = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = train_ds.map(
    lambda batch: tokenize_function(batch, tokenizer),
    batched=True,
    batch_size=1000,
)
val_ds = val_ds.map(
    lambda batch: tokenize_function(batch, tokenizer),
    batched=True,
    batch_size=1000,
)

keep_columns = ["input_ids", "attention_mask", "label"]
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_columns])
val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_columns])

train_ds.set_format("torch")
val_ds.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=200,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print("\nEval metrics:", metrics)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Labels:", label2id)
