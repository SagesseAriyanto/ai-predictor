import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("./ai_data.csv")

# TODO: Model to determine AI company category based on description
data_categories = data.dropna(
    subset=["Description", "Category"]
)  # drop rows with missing Description or Category

# Split data into features X and target y (y = f(X))D
X_desc = data_categories["Description"]  # feature column
y_category = data_categories["Category"]  # target column

# split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_desc, y_category, test_size=0.2, random_state=42
)

# ML models learn better with numerical data
vectorizer = TfidfVectorizer(
    max_features=1000, stop_words="english"
)  # limit to top 1000 features with common English stop words removed

# Fit training data and transform it (learn + convert)
X_train_vec = vectorizer.fit_transform(X_train)

# Transform testing data (convert only)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# # Save the trained model and vectorizer (wb = write binary)
# pickle.dump(model, open("./Models/model_category.pkl", "wb"))
# pickle.dump(vectorizer, open("./Models/vectorizer_category.pkl", "wb"))

# TODO: Models to determine success and upvotes for AI companies based on category

# Create 'Success' column based on median upvotes per category
data.dropna(inplace=True)       # drop NaN rows
data['Category_median'] = data.groupby('Category')['Upvotes'].transform('median')
data['Success'] = (data['Upvotes'] >= data['Category_median']).astype(int)
