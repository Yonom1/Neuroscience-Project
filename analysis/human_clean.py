import pandas as pd

data = pd.read_csv("analysis/2025-12-03T12-23_export.csv")

TN = {}
TP = {}
FN = {}
FP = {}

for line in data.itertuples():
    dataset = line.Condition
    true = line.true
    pred = line.pred

    if dataset not in TN:
        TN[dataset] = 0
        TP[dataset] = 0
        FN[dataset] = 0
        FP[dataset] = 0

    if true == 0 and pred == 0:
        TN[dataset] += 1
    elif true == 1 and pred == 1:
        TP[dataset] += 1
    elif true == 1 and pred == 0:
        FN[dataset] += 1
    elif true == 0 and pred == 1:
        FP[dataset] += 1


def dataset_map(origin: str) -> str:
    if origin == "Original":
        return "dataset"
    else:
        return f"dataset_variants/{origin.lower().replace('l', 'level_')}"


summary_rows = []
for dataset in TN.keys():
    summary_rows.append(
        {
            "Dataset": dataset_map(dataset),
            "Model": "human",
            "TN": TN[dataset],
            "FP": FP[dataset],
            "FN": FN[dataset],
            "TP": TP[dataset],
        }
    )

summary_rows_df = pd.DataFrame(summary_rows)
summary_rows_df.to_csv("analysis/human.csv", index=False)
