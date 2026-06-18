import pandas as pd

df = pd.read_json("prediction_task2.jsonl", lines=True)
df.to_csv("submission_task2.csv", index=False)