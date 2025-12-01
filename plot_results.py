import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Read the CSV file
df = pd.read_csv('experiment_results.csv')

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

# Noise: 0.0 (Original), 0.05, 0.20, 0.35, 0.5, 0.75
noise_map = {
    0: 0.0,
    1: 0.05,
    2: 0.20,
    3: 0.35,
    4: 0.5,
    5: 0.75
}

# Process the dataframe to separate Contrast and Noise experiments
contrast_data = []
noise_data = []

for index, row in df.iterrows():
    dataset = row['Dataset']
    model = row['Model']
    accuracy = row['Accuracy']
    
    if dataset == 'dataset':
        # Original dataset belongs to both (Level 0)
        contrast_data.append({'Model': model, 'Level': 0, 'Value': contrast_map[0], 'Accuracy': accuracy})
        noise_data.append({'Model': model, 'Level': 0, 'Value': noise_map[0], 'Accuracy': accuracy})
        
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

df_contrast = pd.DataFrame(contrast_data)
df_noise = pd.DataFrame(noise_data)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Accuracy vs Contrast Level (Value)
sns.lineplot(data=df_contrast, x='Value', y='Accuracy', hue='Model', marker='o', ax=axes[0])
axes[0].set_title('Accuracy vs Contrast Level')
axes[0].set_xlabel('Contrast (1.0 = Original, Lower is worse)')
axes[0].set_ylabel('Accuracy')
axes[0].invert_xaxis() # Invert x-axis because lower contrast is "higher level" of degradation
axes[0].grid(True)

# Plot 2: Accuracy vs Noise Level (Value)
sns.lineplot(data=df_noise, x='Value', y='Accuracy', hue='Model', marker='o', ax=axes[1])
axes[1].set_title('Accuracy vs Noise Level')
axes[1].set_xlabel('Noise Width (0.0 = Original, Higher is worse)')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('accuracy_plot.png')
print("Plot saved to accuracy_plot.png")
