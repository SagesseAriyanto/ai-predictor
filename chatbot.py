from google import genai
from dotenv import load_dotenv
import pandas as pd
import os


def get_resp(prompt):
    load_dotenv()

    # Read data from CSV
    df = pd.read_csv('./ai_data.csv')
    print(df.head())
    print(df.info())
    print(df['Price'].value_counts())
    # client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    # response = client.models.generate_content(
    #     model="gemini-2.5-flash", contents=prompt
    # )

    # return response.text
    
get_resp("text")
