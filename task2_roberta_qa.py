import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)

# Load data
train = pd.read_json("train.jsonl", lines=True)

examples = []

for _, row in train.iterrows():

    paragraphs = row["targetParagraphs"]

    for pos in row["spoilerPositions"]:

        # skip malformed samples
        if len(pos) < 2:
            continue

        start_para = pos[0][0]
        start_char = pos[0][1]

        end_para = pos[1][0]
        end_char = pos[1][1]

        # skip cross-paragraph samples
        if start_para != end_para:
            continue

        if start_para >= len(paragraphs):
            continue

        context = paragraphs[start_para]

        examples.append({
            "question": str(row["postText"][0]),
            "context": context,
            "answer_text": context[start_char:end_char],
            "answer_start": start_char
        })

print("Training examples:", len(examples))

dataset = Dataset.from_pandas(
    pd.DataFrame(examples)
)

# Train / Val split
dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Model
model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

def preprocess(examples):

    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        padding="max_length",
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(
        tokenized["offset_mapping"]
    ):

        answer_start = examples["answer_start"][i]

        answer_end = (
            answer_start
            + len(examples["answer_text"][i])
        )

        sequence_ids = tokenized.sequence_ids(i)

        start_token = 0
        end_token = 0

        for idx, (offset_start, offset_end) in enumerate(offsets):

            if sequence_ids[idx] != 1:
                continue

            if (
                offset_start <= answer_start
                < offset_end
            ):
                start_token = idx

            if (
                offset_start < answer_end
                <= offset_end
            ):
                end_token = idx

        start_positions.append(
            start_token
        )

        end_positions.append(
            end_token
        )

    tokenized["start_positions"] = (
        start_positions
    )

    tokenized["end_positions"] = (
        end_positions
    )

    tokenized.pop("offset_mapping")

    return tokenized

train_dataset = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=train_dataset.column_names
)

val_dataset = val_dataset.map(
    preprocess,
    batched=True,
    remove_columns=val_dataset.column_names
)

model = AutoModelForQuestionAnswering.from_pretrained(
    model_name
)

training_args = TrainingArguments(
    output_dir="./task2_qa_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

print("Training complete.")