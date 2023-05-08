import pandas as pd
from preprocessing import preprocess
import os
from pathlib import Path
import pickle


main= Path(__file__).parent.parent
model_path =os.path.join(main,"models","model.pkl")
with open(model_path,"rb") as f:
    model=pickle.load(f)
    
def predictor(text):
        data = {'text':  [text]}
        new_df = pd.DataFrame(data)
        final_df = preprocess(new_df)
        arr = final_df.to_numpy()
        result=model.predict(arr)
        return result
    
text=input("enter thought : ")
result=predictor(text)

if result==[1]:
        print("Suicidal")
else:
        print("Not Suicidal")