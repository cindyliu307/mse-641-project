import json

with open("test.jsonl", "r", encoding="utf-8") as f:
    first_line = f.readline()

print(first_line)