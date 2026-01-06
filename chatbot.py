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

    return df, categories_list, prices_list

@st.cache_data
def get_available_models():
    models_file = "gemini_models.json"

    if os.path.exists(models_file):
        with open(models_file, "r") as file:
            models_list = json.load(file)
        return models_list
    models_list = []
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    for model in client.models.list():
        if "generateContent" in model.supported_actions:
            model_name = model.name.replace("models/", "")
            models_list.append(model_name)

    # Save models to a JSON file
    with open(models_file, "w") as file:
        json.dump(models_list, file)
    return models_list


@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_resp(chat_history: list) -> str:
    # Load data and initialize client
    df, categories_list, prices_list = load_chatbot_data()
    client = get_genai_client()
    models = get_available_models()

    if df.empty: return "Error: Database not loaded."
    if not chat_history: return "Hello! How can I help you?"

    # Extract the latest question and past history
    current_question = chat_history[-1]["content"]
    past_history = chat_history[:-1][-5:]

    history_text = ""
    for msg in past_history:
        role = "User" if msg["role"] == "user" else "AI"
        content = str(msg["content"])[:200]
        history_text += f"{role}: {content}\n"

    # Hybrid prompt with context
    prompt = f"""
    I have a pandas DataFrame 'df' with AI tools.
    Columns:
    - Name (str): Tool name
    - Category (str): Valid values: {categories_list}
    - Price (str): Valid values: {prices_list}
    - Upvotes (int): Number of upvotes
    - Link (str): URL
    - Description (str)

    --- HISTORY ---
    {history_text}
    ----------------------------
    
    CURRENT QUESTION: "{current_question}"

    INSTRUCTIONS
    1. INTENT CHECK:
    - if the user greets or asks general concepts (e.g., "What is LLM?"), reply with a brief short simple text explanation without code.
    - if the user asks for tool recommendations, counts, or stats, write PYTHON CODE.
    2. CODING RULES (For Data Queries):
    - Return ONLY the python code string. No markdown, explanations, or additional text.
    - Fuzzy Name Search: ALWAYS use 'df[df['Name'].str.contains("search_term", case=False)]'.
    - Category Search: Check if the category exists in Valid Values. (same for price)
    - Sorting: Use '.sort_values(by="Upvotes", ascending=False)' for "best" or "popular" tools.
    - Columns: Select relevant columns (.e.g., [['Name', 'Link', 'Price']]) to keep output clean.
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

    code = (
        response.text.replace("```python", "")
        .replace("```", "")
        .replace("`", "")
        .strip()
    )
    print(f"Generated Code:\n{code}\n")
    if code.startswith("df") or "pd." in code or "df[" in code:
        try:
            if code.startswith("print("):
                code = code[6:-1]

            # Run the generated code
            result = eval(code)

            # Format the result
            if isinstance(result, pd.DataFrame):
                return result.to_markdown(index=False)
            
            elif isinstance(result, pd.Series):
                return result.to_markdown()
            
            else:
                return str(result)
            
        except:
            return f"An unexpected error occured. Try rephrasing your question."
    else:
        return code

get_resp([{"role": "user", "content": "give me the most popular ai tool"}])