from ucimlrepo import fetch_ucirepo

ai4i = fetch_ucirepo(id=601)
df = ai4i.data.original
df.to_csv("data/raw/ai4i2020.csv", index=False)
print(f"Saved dataset: {df.shape[0]} rows, {df.shape[1]} columns")