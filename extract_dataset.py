import preprocess
import pandas as pd

filename = 'dataset_related.csv'

df = pd.read_csv(filename)
df_extract = df.loc[:, ['tweet', 'class']]
df_extract = df_extract.dropna()
df_extract = df_extract.drop_duplicates()