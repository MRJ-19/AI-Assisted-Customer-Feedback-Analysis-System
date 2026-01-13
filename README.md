📊 AI-Assisted Customer Feedback Analysis System
🌟 Project Overview
This system provides a high-performance bridge between traditional Machine Learning and Generative AI. It classifies sentiment using a specialized Scikit-learn pipeline and leverages Gemma 3 to explain the nuances of customer feedback.

📂 File-by-File Guide

1. train_from_db.py — The Data Processor
   
Introduction: This is your primary production training script. It connects to an IMDB.db SQLite database to extract reviews and ratings.
Instructions:
Ensure IMDB.db is in the root directory.
Run python train_from_db.py to normalize data and train a TF-IDF + Logistic Regression pipeline.
It saves the final model as sentiment_model.pkl.

2. app.py — The Command Center (UI)
   
Introduction: A Streamlit dashboard that offers an interactive interface for both real-time analysis and database-wide reporting.
Instructions:
Launch using streamlit run app.py.
Single Analysis: Paste text to see a color-coded sentiment result, confidence progress bar, and AI-generated word cloud.
Database Overview: Click the sidebar button to run a batch analysis on 5,450+ reviews with Plotly visualizations.

3. feedback_api.py — The REST Service
   
Introduction: A FastAPI backend that exposes the sentiment analysis pipeline for external applications and programmatic access.
Instructions:
Start the server using uvicorn feedback_api:app --reload.
Send a POST request to /analyze with a JSON body: {"text": "your feedback here"}.
It returns the sentiment, a 10-word AI summary, and the raw text.

4. train_model.py — The Prototype Script
   
Introduction: A lightweight script used for testing the pipeline logic with a small, hardcoded dataset.
Instructions:
Use this for quick debugging or to verify the environment setup without needing the full SQLite database.
It demonstrates how "positive", "negative", and "neutral" labels are initially handled.
