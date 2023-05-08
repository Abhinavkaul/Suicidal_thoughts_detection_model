import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("suicide2.csv")
new_df=df.copy()
new_df.isnull().sum()
new_df["label"] = new_df["label"].map({"depression": 0, 
                                    "SuicideWatch": 1})
print('count:-')
print(new_df['label'].value_counts())
print()
print("percentage:-")
print((new_df['label'].value_counts()/new_df['label'].count())*100)
new_df['label'].value_counts().plot(kind='bar')
