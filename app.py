import streamlit as st
import pickle

model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

st.title("📧 Spam Email Classifier")

text = st.text_area("Enter Email Text")

if st.button("Predict"):
    vec = vectorizer.transform([text])
    result = model.predict(vec)

    if result[0] == 1:
        st.error("Spam Email 🚨")
    else:
        st.success("Not Spam ✅")
