import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load the big data
# Replace 'large_dataset.csv' with your file name
df = pd.read_csv('large_dataset.csv') 

# 2. Basic Cleaning
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

# 3. Split: 80% to learn, 20% to test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

# 4. The Pipeline (Vectorize -> Train)
# ngram_range=(1,2) lets the model understand "not good" vs "good"
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

# 5. Train
print("🚀 Training on large dataset... this may take a minute.")
model_pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save
joblib.dump(model_pipeline, "pro_sentiment_model.pkl")