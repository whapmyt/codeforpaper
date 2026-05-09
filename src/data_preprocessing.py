
import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['price','positive_ratings','negative_ratings'])
    df['success'] = (df['positive_ratings'] > 1000).astype(int)
    return df
