import pandas as pd
import os

def calculate_accuracy():
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

    required_columns = ['True', 'Pred', 'Condition']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain columns: {required_columns}")
        return

    total_accuracy = (df['True'] == df['Pred']).mean()
    print(f"Overall Accuracy: {total_accuracy:.2%}\n")

    print(f"{'Condition':<20} | {'Accuracy':<10} | {'Count':<10}")
    print("-" * 46)
    
    conditions = sorted(df['Condition'].unique())
    
    results = []

    for condition in conditions:
        subset = df[df['Condition'] == condition]
        accuracy = (subset['True'] == subset['Pred']).mean()
        count = len(subset)
        
        print(f"{condition:<20} | {accuracy:.2%}    | {count:<10}")
        
        results.append({
            'Condition': condition,
            'Accuracy': accuracy,
            'Count': count
        })


if __name__ == "__main__":
    calculate_accuracy()
