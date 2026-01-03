import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("./ai_data.csv")

@st.cache_resource(show_spinner=False)
def load_models():
    category_model = pickle.load(open("./Models/model_category.pkl", "rb"))
    category_vectorizer = pickle.load(open("./Models/vectorizer_category.pkl", "rb"))
    success_model = pickle.load(open("./Models/model_success.pkl", "rb"))
    description_vectorizer = pickle.load(
        open("./Models/vectorizer_description.pkl", "rb")
    )
    category_encoder = pickle.load(open("./Models/label_encoder_category.pkl", "rb"))
    price_encoder = pickle.load(open("./Models/label_encoder_price.pkl", "rb"))
    return (category_model, category_vectorizer, success_model, description_vectorizer,
            category_encoder, price_encoder)


def get_company_counts(category) -> tuple:
    df = load_data()
    return int((df['Category'] == category).sum())

def get_rank(category):
    df = load_data()
    category_counts = df.groupby("Category", as_index=False)["Upvotes"].sum().sort_values(by="Upvotes", ascending=False).reset_index(drop=True)
    rank = category_counts.loc[category_counts['Category'] == category].index[0] + 1
    return (rank, len(category_counts))

def get_median_size() -> float:
    df = load_data()
    return df["Category"].value_counts().median()

def predict_category(desciption):
    category_model, category_vectorizer, _, _, _, _ = load_models()

    # Vectorize the input description (convert to numerical data)
    desc_vec = category_vectorizer.transform([desciption])
    # Predict the category
    category_prediction = category_model.predict(desc_vec)[0]
    return category_prediction

def predict_success(description, category, price):
    _, _, success_model, description_vectorizer, category_encoder, price_encoder = load_models()
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


# Streamlit title and subtitle
st.markdown(
    "<div style='text-align: center; font-size: 4rem; margin-bottom: -2.5rem;'>🤖</div>",
    unsafe_allow_html=True,
)
st.title("ValidAI", anchor=False, text_alignment="center")
st.markdown(
    "<div style='text-align: center; margin-top: -1rem; margin-bottom: 0.7rem; font-size: 1.3rem; font-weight: 660;'>Validate your vision against <span style='background-color: #DDD6FE; color: #374151; padding: 2.5px 7px; border-radius: 5px; font-weight: 700; margin-left: 3px;'>4000 existing AI tools</span></div>",
    unsafe_allow_html=True,
)

# Tabs for different functionalities
validate_tab, chat_tab, dataset_tab = st.tabs(["Validate", "Chat", "Dataset"])

# Validate Tab
with validate_tab:
    # Description Input
    description = st.text_area(
        "",
        height=150,
        max_chars=300,
        placeholder="Briefly describe your AI tool",
        label_visibility="collapsed",
    )
    # Custom CSS for Text Area
    st.markdown(
        """
    <style>
    textarea {
    resize: none !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if description:
        with st.spinner("Analyzing..."):
            category = predict_category(description)
            companies_count = get_company_counts(category)
            price_types = ["Free", "Freemium", "Paid"]
            success_scores = [predict_success(description, category, price) for price in price_types]
            success_avg = round(sum(success_scores) / len(success_scores))
            col1, col2, col3 = st.columns(3)
            # 1. MARKET (Simple & Clean)
            col1.metric(
                label="Market",
                value=f"{category}",
                border=True,
                help="The identified category based on your description.",
            )

            # 2. COMPETITION (Uses 'inverse' delta)
            # Logic: If competition is high (>500), show it as 'red' (inverse), otherwise green.
            col2.metric(
                label="Competition",
                value=f"{companies_count}",
                border=True,
                delta="High Saturation" if companies_count > 500 else "Low Saturation",
                delta_color="inverse",  # Red arrow if high, Green arrow if low
                help="Number of existing tools in this specific market.",
            )

            # 3. SCORE (Uses 'chart_data' from your list!)
            # We use your 'success_scores' list to draw a mini trend line inside the card.
            col3.metric(
                label="Score",
                value=f"{success_avg}%",
                border=True,
                delta=f"{success_avg - 50}% vs Avg",  # Shows if you are above/below average (50%)
                chart_data=success_scores,  # <--- Uses your list [Free, Freemium, Paid] scores
                help="Success probability across different pricing models (Free vs Paid).",
            )


# Chat Tab
with chat_tab:
    st.info("Coming Soon!")

# Dataset Tab
with dataset_tab:
    st.info("Coming Soon!")
    # if description:
    #     category = predict_category(description)
    #     st.success(f"Category: {category}")
    #     price_types = ["Free", "Freemium", "Paid"]
    #     success_probs = [predict_success(description, category, price).join(" %") for price in price_types]
    #     col1, col2, col3 = st.columns(3)
    #     col1.metric(label="Free", value=success_probs[0])
