import os

import pandas as pd


def calculate_confusion_matrix():
    csv_path = "human_xr/classification_results_web_v2.csv"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("CSV file is empty.")
        return

    # 检查必要的列是否存在
    required_columns = ["True", "Pred", "Condition"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain columns: {required_columns}")
        return

    # 按 Condition 分组计算混淆矩阵
    print(
        f"{'Condition':<25} | {'TN':<6} | {'FP':<6} | {'FN':<6} | {'TP':<6} | {'Accuracy':<10}"
    )
    print("-" * 80)

    # 获取所有唯一的 Condition 并排序
    conditions = sorted(df["Condition"].unique())

    results = []

    for condition in conditions:
        subset = df[df["Condition"] == condition]

        # 计算混淆矩阵: 0=Persian, 1=Siamese
        TN = ((subset["True"] == 0) & (subset["Pred"] == 0)).sum()  # Persian correct
        FP = ((subset["True"] == 0) & (subset["Pred"] == 1)).sum()  # Persian -> Siamese
        FN = ((subset["True"] == 1) & (subset["Pred"] == 0)).sum()  # Siamese -> Persian
        TP = ((subset["True"] == 1) & (subset["Pred"] == 1)).sum()  # Siamese correct

        accuracy = (TN + TP) / (TN + FP + FN + TP) if (TN + FP + FN + TP) > 0 else 0

        print(
            f"{condition:<25} | {TN:<6} | {FP:<6} | {FN:<6} | {TP:<6} | {accuracy:.2%}"
        )

        results.append(
            {
                "Dataset": condition,
                "Model": "human",
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "TP": TP,
            }
        )

    # 保存混淆矩阵结果到CSV文件
    result_df = pd.DataFrame(results)
    output_path = "human_xr/human_confusion_matrix.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\nConfusion matrix saved to {output_path}")


if __name__ == "__main__":
    calculate_confusion_matrix()
