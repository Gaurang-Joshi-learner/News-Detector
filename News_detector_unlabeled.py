import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
df=pd.DataFrame()
df=pd.read_csv('unlabeled_news.csv')
model=joblib.load('news_model.pkl')
print(df.columns)
x=df['text']
print(df['text'].unique())
tfidf = joblib.load('tfidf_vectorizer.pkl')
X=tfidf.transform(df['text'])
predictions=model.predict(X)
df['predictionsl']=predictions
df['predictionsl']=df['predictionsl'].map({1:'REAL',0:'FAKE'})
print(df[['text','predictionsl']])
