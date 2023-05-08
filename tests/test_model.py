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
  
def test_sample1():
    text="Today, I felt good in the morning; everything was good, but in the evening, it rained, and as a result, I got stuck in traffic. My life sucks; I should end it; I should kill myself."
    result=predictor(text)
    assert result == 1 , "Should be 'not suicidal'"
        
def test_sample2():
    text="Today I felt good in the morning, everything was good, but in the evening, it rained, and as a result, I got stuck in the traffic; my life sucks"
    result=predictor(text)
    assert result == 0 , "Should be 'suicidal'"