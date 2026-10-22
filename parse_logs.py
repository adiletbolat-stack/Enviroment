
import pandas as pd
import json

# Path to the raw JSONL file
file_path = "/data/raw/test_run.jsonl"

# Load records from the JSONL file
data = []

with open(file_path, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Invalid JSON on line {line_number}")

# Convert the records into a Pandas DataFrame
df = pd.DataFrame(data)

# Inspect the loaded data
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())
