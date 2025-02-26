import pandas as pd
import os
import json
from collections import defaultdict

# Define the directory containing the difficulty configurations
base_dir = 'data/kk/instruct'
difficulty_dirs = ['3ppl', '4ppl', '5ppl', '6ppl', '7ppl']

# Dictionary to store the results
results = {
    'problem_counts': {},
    'attribute_comparison': {}
}

# Dictionary to store dataframes for attribute comparison
all_train_dfs = {}
all_test_dfs = {}

# Analyze each difficulty configuration
for difficulty in difficulty_dirs:
    dir_path = os.path.join(base_dir, difficulty)
    train_path = os.path.join(dir_path, 'train.parquet')
    test_path = os.path.join(dir_path, 'test.parquet')
    
    # Check if the files exist
    if os.path.exists(train_path) and os.path.exists(test_path):
        # Load the parquet files
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # Store the dataframes for later comparison
        all_train_dfs[difficulty] = train_df
        all_test_dfs[difficulty] = test_df
        
        # Count the number of problems
        results['problem_counts'][difficulty] = {
            'train': len(train_df),
            'test': len(test_df),
            'total': len(train_df) + len(test_df)
        }

# Compare attributes across different difficulty configurations
def compare_attributes(dfs_dict):
    if not dfs_dict:
        return "No dataframes to compare"
    
    # Get the first dataframe as reference
    first_key = list(dfs_dict.keys())[0]
    first_df = dfs_dict[first_key]
    
    # Get the columns (attributes) of the first dataframe
    attributes = first_df.columns.tolist()
    
    # Check if all dataframes have the same attributes
    all_same_attributes = True
    for difficulty, df in dfs_dict.items():
        if set(df.columns) != set(attributes):
            all_same_attributes = False
            break
    
    return {
        'all_same_attributes': all_same_attributes,
        'attributes': attributes
    }

# Compare attributes for train and test datasets
results['attribute_comparison']['train'] = compare_attributes(all_train_dfs)
results['attribute_comparison']['test'] = compare_attributes(all_test_dfs)

# Generate markdown content
markdown_content = "# Analysis of Problem Datasets Across Difficulty Configurations\n\n"

# Problem counts section
markdown_content += "## Problem Counts\n\n"
markdown_content += "| Difficulty | Train Problems | Test Problems | Total Problems |\n"
markdown_content += "|------------|---------------|--------------|---------------|\n"

for difficulty, counts in results['problem_counts'].items():
    markdown_content += f"| {difficulty} | {counts['train']} | {counts['test']} | {counts['total']} |\n"

# Attribute comparison section
markdown_content += "\n## Attribute Comparison\n\n"

# Train dataset attributes
train_attr_result = results['attribute_comparison']['train']
markdown_content += "### Train Dataset Attributes\n\n"
if train_attr_result['all_same_attributes']:
    markdown_content += "All train datasets across different difficulty configurations have the same attributes.\n\n"
    markdown_content += "**Attributes:**\n\n"
    for attr in train_attr_result['attributes']:
        markdown_content += f"- {attr}\n"
else:
    markdown_content += "Train datasets have different attributes across difficulty configurations.\n"

# Test dataset attributes
test_attr_result = results['attribute_comparison']['test']
markdown_content += "\n### Test Dataset Attributes\n\n"
if test_attr_result['all_same_attributes']:
    markdown_content += "All test datasets across different difficulty configurations have the same attributes.\n\n"
    markdown_content += "**Attributes:**\n\n"
    for attr in test_attr_result['attributes']:
        markdown_content += f"- {attr}\n"
else:
    markdown_content += "Test datasets have different attributes across difficulty configurations.\n"

# Write the markdown content to a file
with open(os.path.join(base_dir, 'dataset_analysis.md'), 'w') as f:
    f.write(markdown_content)

print(f"Analysis complete. Results written to {os.path.join(base_dir, 'dataset_analysis.md')}")
