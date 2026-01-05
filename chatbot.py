from google import genai
from dotenv import load_dotenv
import pandas as pd
import os


def get_resp(prompt):
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
    print(df.isna().sum())

    # Convert NaN prices to 'N/A'
    df['Price'] = df['Price'].fillna('N/A')

    df.dropna(inplace=True)
    print(df.isna().sum())
    
    # client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    # response = client.models.generate_content(
    #     model="gemini-2.5-flash", contents=prompt
    # )

    # return response.text
    
get_resp("text")
