import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from chatbot import get_resp

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
    category_counts['Rank'] = category_counts.index + 1
    rank = category_counts.loc[category_counts['Category'] == category, 'Rank'].values[0]
    total = len(category_counts)

    # Finding 5-category window
    start = max(1, rank - 2)            
    end = min(total, rank + 2)

    # Check if window is less than 5 and adjust
    count = end - start + 1
    missing = 5 - count
    if missing > 0:
        start = max(1, start - missing)
        end = min(total, end + missing)
    
    # Select and copy the window
    window = category_counts.iloc[start-1:end].copy()
    return (rank, total, window)

def get_median_size(category_list) -> float:
    df = load_data()
    counts = df["Category"].value_counts()
    category_count = [counts.get(cat, 0) for cat in category_list]
    median_size = round(counts.median(), 1)
    return (median_size, category_count)

def get_average_success() -> float:
    df = load_data()
    df.dropna(inplace=True)
    df["Category_median"] = df.groupby("Category")["Upvotes"].transform("median")
    df["Success"] = (df["Upvotes"] >= df["Category_median"]).astype(int)
    return round(df["Success"].mean() * 100, 1)

def get_top_bottom_tools(category):
    df = load_data()
    category_df = df[df['Category'] == category].copy()

    # Top 25% by upvotes
    top_threshold = category_df['Upvotes'].quantile(0.75)
    top_pool = category_df[category_df['Upvotes'] >= top_threshold]
    top_tools = top_pool.sample(n=min(3, len(top_pool)), random_state=None)[
        ["Name", "Price", "Link"]
    ]

    # Bottom 25% by upvotes
    bottom_threshold = category_df['Upvotes'].quantile(0.25)
    bottom_pool = category_df[category_df['Upvotes'] <= bottom_threshold]
    bottom_tools = bottom_pool.sample(n=min(3, len(bottom_pool)), random_state=None)[
        ["Name", "Price", "Link"]
    ]

    # Fill missing prices
    top_tools["Price"] = top_tools["Price"].fillna("N/A")
    bottom_tools["Price"] = bottom_tools["Price"].fillna("N/A")
    return top_tools, bottom_tools

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
        # Prediction and Metrics Display
        with st.spinner("Analyzing..."):

            # Load additional metrics
            category = predict_category(description)
            companies_count = get_company_counts(category)
            price_types = ["Free", "Freemium", "Paid"]
            success_scores = [predict_success(description, category, price) for price in price_types]
            success_avg = round(sum(success_scores) / len(success_scores))

            col1, col2, col3 = st.columns([2.07,1,1])               # Split into 3 columns and adjust spacing

            # Data for first column (Market)
            rank, total_categories, window = get_rank(category)
            rank_text = f"Rank {rank} of {total_categories}"
            if rank <= 5:
                rank_color = "normal"
            elif rank <= 10:
                rank_color = "off"
            else:
                rank_color = "inverse"
            category_list = window['Category'].tolist()
            upvote_list = window['Upvotes'].tolist()
            rank_list = window['Rank'].tolist()

            # Data for second column (Competition)
            median_size, cat_count_list = get_median_size(category_list)
            percent_diff = round(((companies_count - median_size) / median_size) * 100, 1)
            if percent_diff <= -10:
                count_text = f"{percent_diff}% low"
                count_color = "inverse"
            elif percent_diff < 10:
                count_text = f"{percent_diff}% average"
                count_color = "off"
            else:
                count_color = "inverse"
                if percent_diff >= 300:
                    count_text = f"{300}%+ high"
                else:
                    count_text = f"{percent_diff}% high"

            # Data for third column (Score)
            average_total_success = get_average_success()
            success_diff = round(success_avg - average_total_success,1)
            if success_diff >= 10:
                success_text = f"{success_diff}% high"
                success_color = "normal"
            elif success_diff <= -10:
                success_text = f"{success_diff}% low"
                success_color = "normal"
            else:
                success_text = f"{success_diff}% average"
                success_color = "off"

            # Market Metrics
            col1.metric(
                label="Market",
                value=category,
                width="stretch",
                delta_arrow="off",
                border=True,
                help="Category rank by total upvotes (higher = more popular)",
                delta=rank_text,
                delta_color=rank_color,
            )

            # Competition Metrics
            col2.metric(
                label="Competition",
                value=companies_count,
                border=True,
                help="Existing Tools in this category (lower = less saturated)",
                delta=count_text,
                delta_color=count_color,
            )

            # Score Metrics
            col3.metric(
                label="Score",
                value=f"{success_avg}",
                border=True,
                help="Predicted success score out of 100 (higher = better chance)",
                delta=success_text,
                delta_color=success_color,
            )

            # Create chart dataframe with bold labels for predicted category
            labels = []
            for r, c in zip(rank_list, category_list):
                label = f"{r}. {c}"
                if c == category:
                    label = f"<b>{label}</b>"
                labels.append(label)

            chart_df = pd.DataFrame(
                {
                    "Label": labels,
                    "Upvotes": upvote_list,
                    "Competition": cat_count_list,
                }
            )
            # Melt for grouped bars
            chart_melted = chart_df.melt(id_vars="Label", var_name="Metric", value_name="Value")

            fig = px.bar(
                chart_melted,
                x="Label",
                y="Value",
                color="Metric",
                barmode="group",
                log_y=True,  # Log scale for visibility
                color_discrete_map={"Upvotes": "#C4B5FD", "Competition": "#FB923C"},
                hover_data={"Label": False, "Metric": True, "Value": True}
            )
            fig.update_xaxes(
                title_text="",
                tickangle=20,
            )
            fig.update_yaxes(title_text="")
            fig.update_layout(
                height=400,
                title=dict(
                    text="Overview",
                ),
                legend=dict(
                    title="",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )
            st.plotly_chart(fig)

            # Top and Bottom Tools
            top_tools, bottom_tools = get_top_bottom_tools(category)
            with st.expander("Most Trending in Category", expanded=True):
                # Create a list of 3 columns
                cols = st.columns(3)

                # Loop through each row of dataframe (ignore index)
                for counter, (_, row) in enumerate(top_tools.iterrows()):
                    with cols[counter]:
                        with st.container(border=True):
                            st.write(f"**{row['Name']}**")                 
                            st.markdown(f"**`{row['Price']}`**")                # bold monospaced price
                            st.markdown(
                                f'<a href="{row["Link"]}" style="padding: 4px 12px; border: 1px solid #9CA3AF; border-radius: 4px; text-decoration: none; color: #6B7280; font-size: 0.75rem; font-weight:500;">Visit</a>',
                                unsafe_allow_html=True,
                            )

            with st.expander("Least Trending in Category", expanded=True):
                cols = st.columns(3)
                for counter, (_, row) in enumerate(bottom_tools.iterrows()):
                    with cols[counter]:
                        with st.container(border=True):
                            st.write(f"**{row['Name']}**")                 
                            st.markdown(f"**`{row['Price']}`**")
                            st.markdown(
                                f'<a href="{row["Link"]}" style="padding: 4px 12px; border: 1px solid #9CA3AF; border-radius: 4px; text-decoration: none; color: #6B7280; font-size: 0.75rem; font-weight:500;">Visit</a>',
                                unsafe_allow_html=True,
                            )

# Chat Tab
with chat_tab:
    st.title("Your Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    with st.container(border=False, height=420):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything about ValidAI..."):
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = get_resp(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# Dataset Tab
with dataset_tab:
    st.info("Coming Soon!")
