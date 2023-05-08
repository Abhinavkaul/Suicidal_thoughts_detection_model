import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
import pickle

mnb = MultinomialNB()
lrc = LogisticRegression(solver='liblinear', penalty='l1')
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
voting = VotingClassifier(estimators=[('NB', mnb), ('AdaBoost', abc), ('LR', lrc)],voting='soft')

df=pd.read_csv("suicide2.csv")
new_df=df.copy()
new_df["label"] = new_df["label"].map({"depression": 0, 
                                    "SuicideWatch": 1})
final_df = preprocess(new_df)
print("this is final df",final_df)
X_train,X_test,y_train,y_test = train_test_split(final_df.iloc[:,:-1].values,final_df.iloc[:,-1].values,test_size=0.2,random_state=1)
voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
pickle.dump(voting,open('model.pkl','wb'))
