from flask import Flask, request, jsonify
import joblib
from train import preprocess_text
import streamlit as st

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def predict(question):
    #user_question = request.json['question']
    user_question = question #preprocess_text(user_question)
    user_question_vec = vectorizer.transform([user_question])
    answer = model.predict(user_question_vec)[0]
    return answer #jsonify({'answer': answer})

if __name__ == '__main__':
    # while True:
    #     question = input('Ask a question or type exit to quit: ')
    #     if question == 'exit':
    #         break
    #     else:
    #         print(predict(question))
    st.title("FAQ Chatbot")
    user_question = st.text_input('Enter a question')
    user_question_vec = vectorizer.transform([user_question])
    answer = model.predict(user_question_vec)[0]
    st.button('Predict')
    if st.button:
        st.write(answer)
    print(answer)

# 1. Curl
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"question":"offer free samples?"}'

# 2. Using Postman
# Open Postman and create a new request.
# Set the request type to POST.
# Set the URL to http://127.0.0.1:5000/predict.
# Go to the "Body" tab, select raw and JSON, and enter the following JSON:
#     {
#     "question": "What is your return policy?"
#     }

# 3. Using Python requests module
# import requests
# import json
# reqUrl = "http://127.0.0.1:5000/predict"
# headersList = {
#  "Content-Type": "application/json" 
# }
# payload = json.dumps({
#     "question": "offer samples?"
# })
# response = requests.request("POST", reqUrl, data=payload,  headers=headersList)
# print(response.text)