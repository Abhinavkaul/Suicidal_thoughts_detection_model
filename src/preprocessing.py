from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
import pickle

class Text_preprocessing:
    def preprocess(self,q):
        
        q = str(q).lower().strip()
        
        # Replace certain special characters
        q = q.replace('%', ' percentage ')
        q = q.replace('$', ' dollar ')
        q = q.replace('₹', ' rupee ')
        q = q.replace('€', ' euro ')
        q = q.replace('@', ' at the rate ')
        
        # The pattern '[math]' appears around 900 times in the whole dataset.
        q = q.replace('[math]', '')
        
        # Replacing some numbers with string
        q = q.replace(',000,000,000 ', 'billion ')
        q = q.replace(',000,000 ', 'million ')
        q = q.replace(',000 ', 'thousand ')
        q = re.sub(r'([0-9]+)000000000', r'\1billion', q)
        q = re.sub(r'([0-9]+)000000', r'\1million', q)
        q = re.sub(r'([0-9]+)000', r'\1thousand', q)
        
        # Decontracting words
        # https://stackoverflow.com/a/19794953
        contractions = { 
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
        }
        
        q_decontracted = []

        for word in q.split():
            if word in contractions:
                word = contractions[word]

            q_decontracted.append(word)

        q = ' '.join(q_decontracted)
        q = q.replace("'ve", " have")
        q = q.replace("n't", " not")
        q = q.replace("'re", " are")
        q = q.replace("'ll", " will")
        
        # Removing HTML tags
        patt=re.compile('<.*?>')
        q=patt.sub(r'',q)
        
        # Remove punctuations
        pattern = re.compile('\W')
        q = re.sub(pattern, ' ', q).strip()
        
        #stemming:- process of reducing inflection in words to their root form     
        from nltk.stem.porter import PorterStemmer
        ps=PorterStemmer()
        q= " ".join([ps.stem(word) for word in q.split()])
        
        #------------------------------------------------------------------------------------------------------------------
        
        
        #tokenization
            
        text = nltk.word_tokenize(q)
        
        #check if all characters are alpha-numeric
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        
        #removing stopwords
        text = y[:]
        y.clear()
        STOP_WORDS = stopwords.words("english")
        pickle.dump(STOP_WORDS,open('stopwords.pkl','wb'))
        
        for i in text:
            if i not in STOP_WORDS and i not in string.punctuation:
                y.append(i)        
        text = y[:]
        del y
        
        
        A=" ".join(text)
        
        return A
    
p=Text_preprocessing()
def preprocess(new_df):
    new_df['text'] = new_df['text'].apply(p.preprocess)
    
    txt = list(new_df['text'])

    cv = CountVectorizer(max_features=3000)
    txt_arr= cv.fit_transform(txt).toarray()
    pickle.dump(cv,open('cv.pkl','wb'))
    temp_df = pd.DataFrame(txt_arr)
    final_df = pd.concat([temp_df, new_df], axis=1)
    final_df = final_df.drop(columns=['text'])
    
    return final_df