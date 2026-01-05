from google import genai
from dotenv import load_dotenv
import pandas as pd
import os
load_dotenv()

# Read data from CSV
df = pd.read_csv('./ai_data.csv')

# Map redundant price values to 'Free'
price_mapping = {
    "GitHub": "Free",
    "Open Source": "Free",
    "Google Colab": "Free"
}
df['Price'] = df['Price'].replace(price_mapping)

# Convert NaN prices to 'N/A'
df['Price'] = df['Price'].fillna('N/A')
df.dropna(inplace=True)

# Create unique lists from DataFrame columns
categories_list = ", ".join(df['Category'].unique())
prices_list = ", ".join(df['Price'].unique())

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_resp(question: str) -> str:    
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
        model="gemini-2.5-flash", contents=prompt
    )
    code = response.text.replace("```python", "").replace("```", "").replace("`", "").strip()
    print(code)
    try:
        if code.startswith("print("):
            code = code[6:-1]
        return eval(code)
    except Exception as e:
        return f"Sorry, I couldn't find that in the database."
    