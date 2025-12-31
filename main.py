import streamlit as st
import pickle
import numpy as np
import pandas as pd


def predict_category(desciption):
    category_model = pickle.load(open("./Models/model_category.pkl", "rb"))
    category_vectorizer = pickle.load(open("./Models/vectorizer_category.pkl", "rb"))
    # Vectorize the input description (convert to numerical data)
    desc_vec = category_vectorizer.transform([desciption])
    # Predict the category
    category_prediction = category_model.predict(desc_vec)[0]
    return category_prediction


def predict_success(description, category, price):
    # Load models and encoders
    success_model = pickle.load(open("./Models/model_success.pkl", "rb"))
    description_vectorizer = pickle.load(
        open("./Models/vectorizer_description.pkl", "rb")
    )
    category_encoder = pickle.load(open("./Models/label_encoder_category.pkl", "rb"))
    price_encoder = pickle.load(open("./Models/label_encoder_price.pkl", "rb"))

    # Vectorize the input description
    desc_vec = description_vectorizer.transform([description]).toarray()        # convert to numpy array from sparse matrix
    category_enc = category_encoder.transform([category])                       # returns numpy array
    price_enc = price_encoder.transform([price])                                # returns numpy array

    # Combine all features into a single feature set
    features = np.hstack(
        (desc_vec, category_enc.reshape(-1, 1), price_enc.reshape(-1, 1))       # reshape to 2D arrays for hstack
    )
    # Predict probability of success
    success_prob = int(success_model.predict_proba(features)[0][1] * 100)
    return success_prob


st.markdown("<h1 style='text-align: center;'>ValidAI</h1>", unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center;'>Validate your vision against 4,000 existing AI tools</h5>",
    unsafe_allow_html=True,
)
company_name = st.text_input("Enter your AI company name:")
description = st.text_area("Enter a brief description of your AI company:")

if st.button("Predict"):
    if description:
        category = predict_category(description)
        st.success(f"Category: {category}")
        price_types = ["Free", "Freemium", "Paid"]
        success_probs = [predict_success(description, category, price).join(" %") for price in price_types]
        st.table(
            pd.DataFrame(
                [success_probs],
                columns=price_types,
                index=["Success Probability"],
            )
        )
