from google import genai
from dotenv import load_dotenv
import pandas as pd
import os
import json
import streamlit as st
import numpy as np

load_dotenv()


@st.cache_data
def load_chatbot_data():
    df = pd.read_csv("./ai_data.csv")

    # Map redundant price values to 'Free'
    price_mapping = {"GitHub": "Free", "Open Source": "Free", "Google Colab": "Free"}
    df["Price"] = df["Price"].replace(price_mapping)
    df["Price"] = df["Price"].fillna("N/A")
    df.dropna(inplace=True)

    # Create unique lists from DataFrame columns
    categories_list = ", ".join(df["Category"].unique())
    prices_list = ", ".join(df["Price"].unique())
    columns = ", ".join(df.columns.tolist())
    return f"""
    dataset_columns: {columns}
    dataset_categories: {categories_list}
    dataset_prices: {prices_list}
    total_tools: {len(df)}
    """

@st.cache_data
def get_available_models():
    models_file = "gemini_models.json"

    if os.path.exists(models_file):
        with open(models_file, "r") as file:
            return json.load(file)
    else:
        try:
            models_list = []
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            for model in client.models.list():
                model_name = model.name.replace("models/", "")
                models_list.append(model_name)

                # Save models to a JSON file
            with open(models_file, "w") as file:
                    json.dump(models_list, file)
            return models_list
        except:
            return


@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_resp(chat_history: list) -> str:
    # Load data and initialize client
    context = load_chatbot_data()
    client = get_genai_client()
    models = get_available_models()

    if not chat_history: return "Hello! I can answer general questions or tell you about the ValidAI categories."

    # Extract the latest question and past history
    current_question = chat_history[-1]["content"]
    past_history = chat_history[:-1][-2:]
    history_text = ""
    for msg in past_history:
        role = "User" if msg["role"] == "user" else "AI"
        content = str(msg["content"])[:150]
        history_text += f"{role}: {content}\n"

    # Hybrid prompt with context
    prompt = f"""
    You are the AI Assistant for ValidAI.
    
    BACKGROUND DATA (Use this ONLY for dataset questions):
    {context}

    INSTRUCTIONS:
    1. FIRST, check if the user's question is about the dataset (e.g. "what categories do you have?", "price of tools").
    2. IF YES -> Use the BACKGROUND DATA to answer.
       - LISTS: Mention only 2-3 examples and use "etc." (e.g. "Marketing, Productivity, etc."). Never list them all.
       - MISSING INFO: If asked for specific details about the dataset that are NOT in the background data, simply tell the user to visit the ValidAI Dataset tab to search for it directly.
    3. IF NO (e.g. greetings, "what is AI?") -> Ignore the background data and answer generally.
       - IDENTITY: If asked who you are, identify yourself as the ValidAI assistant.
    4. CONSTRAINT: Keep your response STRICTLY 1-2 sentences maximum.

    CHAT HISTORY:
    {history_text}

    USER QUESTION: "{current_question}"
    """
    response = None
    for model in models:
        try:
            response = client.models.generate_content(
                model=model, contents=prompt
                )
            break
        except:
            continue
    if not response:
        return f"System overloaded. Please try again later."

    return response.text.strip()
