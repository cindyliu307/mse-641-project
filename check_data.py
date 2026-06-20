import pandas as pd

train = pd.read_json("train.jsonl", lines=True)

print(train.columns.tolist())

print("\nTags distribution:")
print(train["tags"].value_counts())