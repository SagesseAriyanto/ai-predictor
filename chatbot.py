from google import genai
from dotenv import load_dotenv
import pandas as pd
import os
import json
import streamlit as st

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


def get_resp(question: str) -> str:
    # Load data and initialize client
    df, categories_list, prices_list = load_chatbot_data()
    client = get_genai_client()
    models = get_available_models()

    prompt = f"""
    I have a pandas DataFrame 'df' with AI tools.
    Columns:
    - Name (str): The tool name
    - Category (str): Valid values: {categories_list}
    - Price (str): Valid values: {prices_list}
    - Upvotes (int): Number of upvotes (higher is more popular)
    - Link (str): The URL
    - Description (str): What it does

    User Question: "{question}"

    Write 1 line of Python code to get the answer.
    Rules:
    1. For Name searches, ALWAYS use 'df[df['Name'].str.contains("search_term", case=False)]'.
    2. For Category searches, check if the category matches one of the Valid Values.
    3. Return ONLY the code string.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp", contents=prompt
    )

    code = (
        response.text.replace("```python", "")
        .replace("```", "")
        .replace("`", "")
        .strip()
    )
    try:
        if code.startswith("print("):
            code = code[6:-1]

        # Run the generated code
        result = eval(code)
        if isinstance(result, pd.DataFrame):
            return result.to_markdown(index=False)
        elif isinstance(result, pd.Series):
            return result.to_markdown()
        else:
            return str(result)
    except Exception as e:
        return f"Sorry, I couldn't find that in the database. Error: {str(e)}"
