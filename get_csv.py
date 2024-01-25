from datasets import load_dataset
import pandas as pd

dataset = load_dataset("climatebert/climate_sentiment")
df = pd.DataFrame(dataset['train'])

df.to_csv('climate_sentiment.csv', index=False)