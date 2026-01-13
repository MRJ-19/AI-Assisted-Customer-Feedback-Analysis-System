import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Sample Training Data
data = [
    ("Great service, very happy!", "positive"),
    ("The product broke after one day.", "negative"),
    ("It was okay, nothing special.", "neutral"),
    ("Worst experience ever, avoid!", "negative"),
    ("I love the new design!", "positive")
]
X, y = zip(*data)

# 2. Create a Pipeline: Feature Extraction (TF-IDF) + Classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# 3. Train and Save
model.fit(X, y)
joblib.dump(model, "sentiment_model.pkl")
print("✅ Model trained and saved as sentiment_model.pkl")