import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Read the CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path1 = os.path.join(script_dir, 'experiment_results.csv')
csv_path2 = os.path.join(script_dir, 'experiment_results2.csv')

df1 = pd.read_csv(csv_path1)
df2 = pd.read_csv(csv_path2)

# Merge dataframes
df = pd.concat([df1, df2], ignore_index=True)

# Calculate Accuracy
df['Accuracy'] = (df['TP'] + df['TN']) / (df['TP'] + df['TN'] + df['FP'] + df['FN'])

# Define the hyperparameter values based on generate_variants.py
# Contrast: 1.0 (Original), 0.9, 0.7, 0.5, 0.3, 0.1
contrast_map = {
    0: 1.0,
    1: 0.9,
    2: 0.7,
    3: 0.5,
    4: 0.3,
    5: 0.1
}

# Noise: 0.0 (Original), 0.05, 0.20, 0.35, 0.5, 0.75, 1.0
noise_map = {
    0: 0.0,
    1: 0.05,
    2: 0.20,
    3: 0.35,
    4: 0.5,
    5: 0.75,
    6: 1.0
}

# Jigsaw: 1 (Original), 2, 4, 8, 16, 32
jigsaw_map = {
    0: 1,
    1: 2,
    2: 4,
    3: 8,
    4: 16,
    5: 32
}

# Eidolon: 0 (Original), 1, 2, 3, 4, 5 (Levels)
eidolon_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5
}

# Process the dataframe to separate Contrast and Noise experiments
contrast_data = []
noise_data = []
jigsaw_data = []
eidolon_data = []

for index, row in df.iterrows():
    dataset = row['Dataset']
    model = row['Model']
    accuracy = row['Accuracy']
    
    if dataset == 'dataset':
        # Original dataset belongs to all (Level 0)
        contrast_data.append({'Model': model, 'Level': 0, 'Value': contrast_map[0], 'Accuracy': accuracy})
        noise_data.append({'Model': model, 'Level': 0, 'Value': noise_map[0], 'Accuracy': accuracy})
        jigsaw_data.append({'Model': model, 'Level': 0, 'Value': jigsaw_map[0], 'Accuracy': accuracy})
        eidolon_data.append({'Model': model, 'Level': 0, 'Value': eidolon_map[0], 'Accuracy': accuracy})
        
    elif 'contrast_level' in dataset:
        # Extract level number
        match = re.search(r'contrast_level_(\d+)', dataset)
        if match:
            level = int(match.group(1))
            contrast_data.append({'Model': model, 'Level': level, 'Value': contrast_map[level], 'Accuracy': accuracy})
            
    elif 'noise_level' in dataset:
        # Extract level number
        match = re.search(r'noise_level_(\d+)', dataset)
        if match:
            level = int(match.group(1))
            noise_data.append({'Model': model, 'Level': level, 'Value': noise_map[level], 'Accuracy': accuracy})

    elif 'jigsaw_level' in dataset:
        match = re.search(r'jigsaw_level_(\d+)', dataset)
        if match:
            level = int(match.group(1))
            jigsaw_data.append({'Model': model, 'Level': level, 'Value': jigsaw_map[level], 'Accuracy': accuracy})

    elif 'eidolon_level' in dataset:
        match = re.search(r'eidolon_level_(\d+)', dataset)
        if match:
            level = int(match.group(1))
            eidolon_data.append({'Model': model, 'Level': level, 'Value': eidolon_map[level], 'Accuracy': accuracy})

df_contrast = pd.DataFrame(contrast_data)
df_noise = pd.DataFrame(noise_data)
df_jigsaw = pd.DataFrame(jigsaw_data)
df_eidolon = pd.DataFrame(eidolon_data)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Accuracy vs Contrast Level (Value)
if not df_contrast.empty:
    sns.lineplot(data=df_contrast, x='Value', y='Accuracy', hue='Model', marker='o', ax=axes[0, 0])
    axes[0, 0].set_title('Accuracy vs Contrast Level')
    axes[0, 0].set_xlabel('Contrast (1.0 = Original, Lower is worse)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].invert_xaxis() 
    axes[0, 0].grid(True)

# Plot 2: Accuracy vs Noise Level (Value)
if not df_noise.empty:
    sns.lineplot(data=df_noise, x='Value', y='Accuracy', hue='Model', marker='o', ax=axes[0, 1])
    axes[0, 1].set_title('Accuracy vs Noise Level')
    axes[0, 1].set_xlabel('Noise Width (0.0 = Original, Higher is worse)')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)

# Plot 3: Accuracy vs Jigsaw Level (Value)
if not df_jigsaw.empty:
    sns.lineplot(data=df_jigsaw, x='Value', y='Accuracy', hue='Model', marker='o', ax=axes[1, 0])
    axes[1, 0].set_title('Accuracy vs Jigsaw Grid Size')
    axes[1, 0].set_xlabel('Grid Size N (1 = Original, Higher is more scrambled)')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_xscale('log', base=2) # Log scale for grid size
    axes[1, 0].grid(True)

# Plot 4: Accuracy vs Eidolon Level (Value)
if not df_eidolon.empty:
    sns.lineplot(data=df_eidolon, x='Value', y='Accuracy', hue='Model', marker='o', ax=axes[1, 1])
    axes[1, 1].set_title('Accuracy vs Eidolon Distortion Level')
    axes[1, 1].set_xlabel('Distortion Level (0 = Original, Higher is worse)')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True)

plt.tight_layout()
save_path = os.path.join(script_dir, 'accuracy_plot.png')
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
