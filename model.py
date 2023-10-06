import streamlit  as st
import pickle
import nltk 
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
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
tfidf=pickle.load(open("models/vectorizr.pkl",'rb'))
model=pickle.load(open("models/spamd.pkl",'rb'))

st.title("Email/SMS Spam Detector")





st.markdown("<h2>Enter the Message</h2>", unsafe_allow_html=True)

input_sms = st.text_area(" ", height=200)


if st.button("Predict"):

    transformed_text=transform_text(input_sms)
    vector_input=tfidf.transform([transformed_text])

    result=model.predict(vector_input)[0]


    if result == 1:
     colored_text = '<h2 style="color: red;">Spam</h2>'
     st.markdown(colored_text, unsafe_allow_html=True)
    elif result==0: 
     colored_text = '<h2 style="color: green;">Not Spam</h2>'
     st.markdown(colored_text, unsafe_allow_html=True)
