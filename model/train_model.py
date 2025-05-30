import kagglehub
import pandas as pd
import os

# Download the latest version
path = kagglehub.dataset_download("subhajournal/phishingemails")

print("Path to dataset files:", path)

# List files in directory to verify exact CSV filename
files = os.listdir(path)
print("Available files:", files)

# Load dataset - using the first CSV file found
csv_files = [f for f in files if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in directory")

csv_path = os.path.join(path, csv_files[0])
print("Using file:", csv_path)

df = pd.read_csv(csv_path)

# Display first rows of dataset
print(df.head())



# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import joblib

# df = pd.read_csv("emails.csv")
# X = df["text"]
# y = df["label"]

# vectorizer = TfidfVectorizer()
# X_vec = vectorizer.fit_transform(X)

# model = LogisticRegression()
# model.fit(X_vec, y)

# joblib.dump(model, "model.pkl")
# joblib.dump(vectorizer, "vectorizer.pkl")
