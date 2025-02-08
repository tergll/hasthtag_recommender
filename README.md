# 📢 AI-Powered Hashtag Recommender

This project uses AI to recommend hashtags for social media posts. It combines a sentence-transformer model for semantic search with OpenAI's GPT-4 to suggest relevant, trending hashtags.

## Features
- Semantic search using FAISS to retrieve similar posts.
- GPT-4 to generate unique and trending hashtag suggestions.
- User-friendly interface built with Streamlit.

## Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/tergll/hasthtag_recommender.git
   cd hasthtag_recommender
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Set your OpenAI API key in an .env file:
   ```bash
    OPENAI_API_KEY=your-api-key
5. Run the app:
  ```bash
  streamlit run app.py
```


## Future Improvements
1. Pull More Data: Enhance recommendations by integrating additional posts from the Bluesky Firehose API.
2. Model Fine-Tuning: Train the sentence-transformer model with domain-specific data for better accuracy.
3. Real-Time Updates: Incorporate live hashtag trends for more dynamic recommendations.


## 📹 Video Tutorial
You can download or preview the tutorial video here: [tutorial.mp4](REC-20250205234359.mp4)


