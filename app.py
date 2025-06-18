import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

tfidf = pickle.load(open ('vectorizer.pkl', 'rb'))
nb = pickle.load(open ('model.pkl', 'rb'))

st.title('SPAM DETECTOR')
input_text = st.text_area('Enter the text')

if st.button('Predict'):
    transformed_text = transform_text(input_text)
    vectorize = tfidf.transform([transformed_text])
    predict = nb.predict(vectorize)[0]
    
    if predict == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")