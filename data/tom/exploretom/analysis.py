

# 'facebook/ExploreToM'

import datasets
import pandas as pd

dataset = datasets.load_dataset('facebook/ExploreToM')

df = dataset['train'].to_pandas()
df.to_excel('explore_tom.xlsx', index=False)
