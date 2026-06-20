import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score


# Load data
train_df = pd.read_json("train.jsonl", lines=True)
val_df = pd.read_json("val.jsonl", lines=True)

# Labels
label_map = {
    "phrase": 0,
    "passage": 1,
    "multi": 2
}

id_to_label = {
    0: "phrase",
    1: "passage",
    2: "multi"
}

train_df["label"] = train_df["tags"].apply(lambda x: label_map[x[0]])
val_df["label"] = val_df["tags"].apply(lambda x: label_map[x[0]])

# Text
test["text"] = (
    test["postText"].astype(str)
    + " [SEP] "
    + test["targetTitle"].astype(str)
)

val_df["text"] = (
    val_df["postText"].astype(str)
    + " [SEP] "
    + val_df["targetTitle"].astype(str)
)

# HuggingFace datasets
train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]]
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]]
)

# Model
model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    return {"macro_f1": f1}

training_args = TrainingArguments(
    output_dir="./bert_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate()

print(results)

# Load test data
test = pd.read_json("test.jsonl", lines=True)

train_df["text"] = (
    train_df["postText"].astype(str)
    + " [SEP] "
    + train_df["targetTitle"].astype(str)
)

test_dataset = Dataset.from_pandas(
    test[["text"]]
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

predictions = trainer.predict(
    test_dataset
)

pred_labels = np.argmax(
    predictions.predictions,
    axis=1
)

pred_tags = [
    id_to_label[x]
    for x in pred_labels
]

submission = pd.DataFrame({
    "id": test["id"],
    "spoilerType": pred_tags
})

submission.to_csv(
    "task1_bert_submission.csv",
    index=False
)

print("Submission file saved.")