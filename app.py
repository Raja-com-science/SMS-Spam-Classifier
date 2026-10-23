import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [
        word for word in tokens
        if word not in stopwords.words("english")
        and word not in string.punctuation
    ]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

@st.cache_resource
def load_models():
    model = joblib.load("spam_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.write("Check whether an SMS message is Spam or Ham.")

message = st.text_area("Enter your message:")

if st.button("Check Message"):
    if message.strip():
        processed_message = transform_text(message)
        message_tfidf = vectorizer.transform([processed_message])
        prediction = model.predict(message_tfidf)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is HAM (Not Spam).")
    else:
        st.warning("Please enter a message.")
