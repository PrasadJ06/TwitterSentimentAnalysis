import streamlit as st
import pickle
import tweepy
import pandas as pd

# Load the trained model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Twitter API credentials (REPLACE with yours)
API_KEY = "9MBdawpBOGHlI0WbDVzGEfDIV"
API_SECRET = "YeCGwLgfVKIDp4pZCv8hpvWw3IUEtfwSI9xCNj6ga823ZaMUwQ"
ACCESS_TOKEN = "1934703197618397185-fQo2xYsdwdn8mTyRJmCVtOs8oNPfy7"
ACCESS_SECRET = "pvMvCNanzOSwdTDAKarRlgEDZnXMubGJX285b8KtqxmVN"

# Authenticate with Twitter
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

# Streamlit page configuration
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")

# Title and description
st.title("Twitter Sentiment Analyzer")
st.markdown("This tool predicts whether tweets express **positive** or **negative** sentiment using a trained ML model.")

# Sentiment prediction logic
def predict_sentiment(text):
    pred = model.predict([text])[0]
    return "Positive" if pred == 1 else "Negative"

# Input method selection
option = st.radio("Choose Input Method", ["Enter a Tweet", "Fetch Tweets by Username"])

# Option 1: User manually enters a tweet
if option == "Enter a Tweet":
    st.subheader("Enter a tweet to analyze sentiment")
    tweet = st.text_area("Tweet text:")

    if st.button("Analyze Sentiment"):
        if tweet.strip() == "":
            st.warning("Please enter a tweet before analyzing.")
        else:
            sentiment = predict_sentiment(tweet)
            st.success(f"Sentiment: {sentiment}")

# Option 2: Fetch tweets using Tweepy
elif option == "Fetch Tweets by Username":
    st.subheader("Analyze sentiment of recent tweets from a user")
    username = st.text_input("Enter Twitter handle (without @):")
    num_tweets = st.slider("Number of recent tweets to fetch", min_value=1, max_value=20, value=5)

    if st.button("Fetch and Analyze Tweets"):
        if not username.strip():
            st.warning("Please enter a valid Twitter username.")
        else:
            try:
                tweets = api.user_timeline(screen_name=username, count=num_tweets, tweet_mode="extended")
                st.success(f"Fetched {len(tweets)} tweets from @{username}")

                for i, tweet in enumerate(tweets, start=1):
                    text = tweet.full_text
                    sentiment = predict_sentiment(text)
                    st.markdown(f"**Tweet {i}: Sentiment — {sentiment}**")
                    st.write(text)
                    st.markdown("---")

            except Exception as e:
                st.error(f"An error occurred while fetching tweets: {e}")
