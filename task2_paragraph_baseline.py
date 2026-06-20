import pandas as pd

# Load test data
test = pd.read_json("test.jsonl", lines=True)

predictions = []

for _, row in test.iterrows():

    paragraphs = row["targetParagraphs"]

    if len(paragraphs) > 0:
        spoiler = str(paragraphs[0]).split(".")[0]
    else:
        spoiler = str(row["targetTitle"])

    if spoiler.strip() == "":
        spoiler = "unknown"

    predictions.append(spoiler)

submission = pd.DataFrame({
    "id": test["id"],
    "spoiler": predictions
})

submission.to_csv(
    "prediction_task2.csv",
    index=False
)

print("Submission file saved.")