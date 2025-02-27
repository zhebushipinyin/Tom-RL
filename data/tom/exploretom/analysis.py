# 'facebook/ExploreToM'

import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Load the Qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M", trust_remote_code=True)

# Define paths
data_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(data_dir, 'train.parquet')
test_path = os.path.join(data_dir, 'test.parquet')

# Function to analyze token lengths
def analyze_token_lengths(file_path, split_name):
    print(f"Analyzing {split_name} dataset...")
    df = pd.read_parquet(file_path)
    
    # Extract the content from the prompt list
    token_lengths = []
    
    for prompt_list in df['prompt']:
        # Each prompt is a list of dictionaries, we want the 'content' of the first dict
        content = prompt_list[0]['content']
        # Tokenize the content
        tokens = tokenizer.encode(content)
        token_lengths.append(len(tokens))
    
    # Convert to numpy array for statistics
    token_lengths = np.array(token_lengths)
    
    # Calculate statistics
    stats = {
        'count': len(token_lengths),
        'min': token_lengths.min(),
        'max': token_lengths.max(),
        'mean': token_lengths.mean(),
        'median': np.median(token_lengths),
        'std': token_lengths.std(),
        'percentiles': {
            '25%': np.percentile(token_lengths, 25),
            '50%': np.percentile(token_lengths, 50),
            '75%': np.percentile(token_lengths, 75),
            '90%': np.percentile(token_lengths, 90),
            '95%': np.percentile(token_lengths, 95),
            '99%': np.percentile(token_lengths, 99),
        }
    }
    
    print(f"\n{split_name} Token Length Statistics:")
    print(f"Count: {stats['count']}")
    print(f"Min: {stats['min']}")
    print(f"Max: {stats['max']}")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Median: {stats['median']}")
    print(f"Std Dev: {stats['std']:.2f}")
    print("\nPercentiles:")
    for percentile, value in stats['percentiles'].items():
        print(f"{percentile}: {value}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, alpha=0.7)
    plt.title(f'Token Length Distribution - {split_name}')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(data_dir, f'{split_name}_token_length_histogram.png'))
    
    return token_lengths, stats

# Analyze both datasets
train_lengths, train_stats = analyze_token_lengths(train_path, 'Train')
test_lengths, test_stats = analyze_token_lengths(test_path, 'Test')

# Compare train and test distributions
plt.figure(figsize=(12, 7))
plt.hist(train_lengths, bins=50, alpha=0.5, label='Train')
plt.hist(test_lengths, bins=50, alpha=0.5, label='Test')
plt.title('Token Length Distribution Comparison')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(data_dir, 'token_length_comparison.png'))

# Save statistics to CSV
stats_df = pd.DataFrame({
    'Metric': ['Count', 'Min', 'Max', 'Mean', 'Median', 'Std Dev', 
               'Percentile 25%', 'Percentile 50%', 'Percentile 75%', 
               'Percentile 90%', 'Percentile 95%', 'Percentile 99%'],
    'Train': [train_stats['count'], train_stats['min'], train_stats['max'], 
              train_stats['mean'], train_stats['median'], train_stats['std'],
              train_stats['percentiles']['25%'], train_stats['percentiles']['50%'], 
              train_stats['percentiles']['75%'], train_stats['percentiles']['90%'],
              train_stats['percentiles']['95%'], train_stats['percentiles']['99%']],
    'Test': [test_stats['count'], test_stats['min'], test_stats['max'], 
             test_stats['mean'], test_stats['median'], test_stats['std'],
             test_stats['percentiles']['25%'], test_stats['percentiles']['50%'], 
             test_stats['percentiles']['75%'], test_stats['percentiles']['90%'],
             test_stats['percentiles']['95%'], test_stats['percentiles']['99%']]
})

stats_df.to_csv(os.path.join(data_dir, 'token_length_statistics.csv'), index=False)
print("\nAnalysis complete. Results saved to CSV and histograms saved as PNG files.")

if __name__ == "__main__":
    print("Running token length analysis for ExploreToM dataset...")
