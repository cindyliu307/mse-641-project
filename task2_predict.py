import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)

# Load model
checkpoint = "./task2_qa_results/checkpoint-1026"

tokenizer = AutoTokenizer.from_pretrained(
    "roberta-base"
)

model = AutoModelForQuestionAnswering.from_pretrained(
    checkpoint
)

model.eval()

# Load test data
test = pd.read_json(
    "test.jsonl",
    lines=True
)

predictions = []

for _, row in test.iterrows():

    question = row["postText"][0]

    paragraphs = row["targetParagraphs"]

    if len(paragraphs) == 0:

        predictions.append(
            row["targetTitle"]
        )

        continue

    best_text = paragraphs[0]
    best_score = -999999

    # search first 10 paragraphs
    for paragraph in paragraphs[:10]:

        inputs = tokenizer(
            question,
            paragraph,
            return_tensors="pt",
            truncation=True,
            max_length=384
        )

        with torch.no_grad():

            outputs = model(**inputs)

        start_idx = torch.argmax(
            outputs.start_logits
        ).item()

        end_idx = torch.argmax(
            outputs.end_logits
        ).item()

        if end_idx < start_idx:
            continue

        score = (
            outputs.start_logits[0][start_idx]
            +
            outputs.end_logits[0][end_idx]
        ).item()

        tokens = inputs["input_ids"][0]

        answer = tokenizer.decode(
            tokens[start_idx:end_idx + 1],
            skip_special_tokens=True
        ).strip()

        if len(answer) == 0:
            continue

        if score > best_score:

            best_score = score
            best_text = answer

    predictions.append(
        best_text
    )

submission = pd.DataFrame({
    "id": test["id"],
    "spoiler": predictions
})

submission.to_csv(
    "prediction_task2.csv",
    index=False
)

print("Submission file saved.")