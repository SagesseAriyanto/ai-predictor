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
    past_history = chat_history[:-1][-3:]

    history_text = ""
    for msg in past_history:
        role = "User" if msg["role"] == "user" else "AI"
        content = str(msg["content"])[:150]
        history_text += f"{role}: {content}\n"

    # Hybrid prompt with context
    prompt = f"""
    Pandas DataFrame 'df' containing AI Tools.
    - Columns: Name, Upvotes, Link, Price, Category
    - Category: {categories_list}
    - Price: {prices_list}

    HISTORY:
    {history_text}
    
    QUESTION: "{current_question}"

    INSTRUCTIONS:
    1. GENERAL: For greetings or concepts, reply with plain text. DO NOT use the word 'df' in text.
    2. DATA QUERY: Write 1 line of PYTHON CODE.
       - "Popular"/"Top"/"Best": Use `df.sort_values(by='Upvotes', ascending=False)`
       - "Find"/"Search": Use `df[df['Name'].str.contains("query", case=False)]`
       - FILTERING: Use `.isin(['Free', 'Paid'])` for prices/categories.
       - FORMATTING: ALWAYS use `.head(n)[['Name', 'Link', 'Upvotes']]` to select columns and limit rows.
       - SAFETY: Return ONLY the code string.
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
        return f"System overloaded. Please try again."

    # Execute generated code
    code = (
        response.text.replace("```python", "")
        .replace("```", "")
        .replace("`", "")
        .strip()
    )
    print(f"Generated Code:\n{code}\n")
    # If response is plain text, return it immediately
    if code.startswith("df") or code.startswith("df[") or "df" in code or "print(" in code or "pd." in code:
        # If response is code, run it
        try:
            if code.startswith("print("):
                code = code[6:-1]

            # Run the generated code
            result = eval(code)

            # DataFrame (Standard Table)
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    return "No results found."
                return result.to_markdown(index=False)

            # Series (Single Column)
            elif isinstance(result, pd.Series):
                return result.to_frame().T.to_markdown(index=False)

            else:
                return str(result)

        except:
            return f"I couldn't process that query. Try rephrasing."
    else:
        return code
