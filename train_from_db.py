import sqlite3
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Connect and Load
conn = sqlite3.connect('IMDB.db')
query = "SELECT review, rating FROM REVIEWS"
df = pd.read_sql_query(query, conn)
conn.close()

# 2. Column Normalization
df.columns = df.columns.str.lower().str.strip()

# 3. Handle Potential Non-Numeric Ratings
# Sometimes SQL "ratings" come in as strings; this forces them to numbers
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating', 'review'])

# 4. Sentiment Logic (Positive if >= 7, else Negative)
df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 7 else 'negative')

print(f"✅ Training on {len(df)} reviews from your IMDB dataset...")

# 5. Build and Train the Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2)
pipeline.fit(X_train, y_train)

# 6. Save the model
joblib.dump(pipeline, "sentiment_model.pkl")
print("🎯 Model saved as sentiment_model.pkl")

def categorize_rating(rating):
    if rating >= 7:
        return 'positive'
    elif rating <= 4:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['rating'].apply(categorize_rating)