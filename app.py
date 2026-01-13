import streamlit as st
import joblib
from langchain_ollama import ChatOllama
import sqlite3
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_sentiment_wordcloud(text, sentiment):
    # Set color based on sentiment
    color_map = {'positive': 'Greens', 'negative': 'Reds', 'neutral': 'YlOrBr'}
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        colormap=color_map.get(sentiment, 'viridis')
    ).generate(text)

    
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig
# Load resources
model = joblib.load("sentiment_model.pkl")
llm = ChatOllama(model="gemma3:4b", temperature=0)

st.title("📊 Customer Sentiment Command Center")

user_input = st.text_area("Paste customer feedback here:")

if st.button("Analyze Sentiment"):
    if user_input:
        # Get the category prediction
        prediction = model.predict([user_input])[0]
        
        # Get the Confidence Score (Probability)
        probabilities = model.predict_proba([user_input])[0]
        max_prob = max(probabilities) * 100 # Convert to percentage
        # Determine bar color based on confidence level
        if max_prob > 85:
                bar_color = "green"  # Very Sure
        elif max_prob > 60:
                bar_color = "orange" # Pretty Sure (This is your 68% case!)
        else:
                bar_color = "red"    # Guessing / Mixed
        
        # Define Colors
        colors = {"positive": "green", "negative": "red", "neutral": "orange"}
        current_color = colors.get(prediction, "blue")

        # UI Display
        st.markdown(f"### Result: :{current_color}[{prediction.upper()}]")
        st.progress(max_prob / 100)
        st.write(f"**Confidence Score:** {max_prob:.1f}%")

        # LLM Insight
        with st.spinner("Gemma 3 analyzing nuance..."):
            # We tell Gemma 3 to be specific about neutrality
            prompt = f"Explain why this movie review is {prediction} (Confidence: {max_prob:.1f}%). Input: {user_input}"
            summary = llm.invoke(prompt).content
            st.info(f"**AI Insight:** {summary}")
            st.pyplot(generate_sentiment_wordcloud(user_input, prediction))  
            


if st.sidebar.button("📊 Analyze Entire Database"):
    # 1. Pull all data from your REVIEWS table
    conn = sqlite3.connect('IMDB.db')
    df_all = pd.read_sql_query("SELECT review FROM REVIEWS", conn)
    conn.close()
    
    # 2. Run your trained model on the whole batch
    with st.spinner("Processing 5,450 reviews..."):
        df_all['sentiment'] = model.predict(df_all['review'])
        
        # 3. Create a Visual Report
        st.divider()
        st.subheader("Database Overview")
        
        col1, col2 = st.columns(2)
        
        # Pie Chart
        fig = px.pie(df_all, names='sentiment', title='Global Sentiment Distribution',
                     color_discrete_map={'positive':'#2ecc71', 'negative':'#e74c3c'})
        col1.plotly_chart(fig)
        
        # Metrics
        pos_count = len(df_all[df_all['sentiment'] == 'positive'])
        neg_count = len(df_all[df_all['sentiment'] == 'negative'])
        col2.metric("Total Reviews", len(df_all))
        col2.metric("Positive", f"{pos_count} ({(pos_count/len(df_all))*100:.1f}%)")
        col2.metric("Negative", f"{neg_count} ({(neg_count/len(df_all))*100:.1f}%)")
    


def generate_sentiment_wordcloud(text, sentiment):
    # Set color based on sentiment
    color_map = {'positive': 'Greens', 'negative': 'Reds', 'neutral': 'YlOrBr'}
    
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        colormap=color_map.get(sentiment, 'viridis')
    ).generate(text)
    
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig
