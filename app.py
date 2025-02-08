import os
import re
import numpy as np
import pandas as pd
import faiss
import openai
import streamlit as st
from sentence_transformers import SentenceTransformer

# --- Streamlit App Title ---
st.title("📢 AI-Powered Hashtag Recommender")

# --- Load OpenAI API Key Securely ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("⚠ OpenAI API key is missing. Set it as an environment variable.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Define File Paths ---
BASE_DIR = os.path.expanduser("~/Downloads/hashtag_rec_project")
DATA_FILE = os.path.join(BASE_DIR, "bluesky_posts.csv")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.bin")

# --- Load Sentence Transformer ---
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# --- Load Dataset ---
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["post_text", "hashtags"])
    st.warning("⚠ No dataset found. Upload 'bluesky_posts.csv' to continue.")

# --- Initialize FAISS Index ---
embedding_dim = 768
faiss_index = faiss.IndexFlatL2(embedding_dim)

if os.path.exists(FAISS_INDEX_FILE):
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
elif not df.empty:
    st.warning("⚠ FAISS index not found. Consider generating one.")

# --- Function: Generate Embeddings ---
def generate_embedding(text):
    """Converts text into an embedding vector."""
    return embedding_model.encode([text], convert_to_numpy=True)

# --- Function: Recommend Hashtags ---
def recommend_hashtags(new_post, top_k=5):
    """Generates relevant hashtags based on similar posts using FAISS and GPT-4."""
    
    if df.empty or faiss_index.ntotal == 0:
        st.warning("⚠ No posts available for recommendations.")
        return ""

    new_embedding = generate_embedding(new_post)

    # Ensure FAISS has data before searching
    if faiss_index.ntotal > 0:
        _, indices = faiss_index.search(new_embedding, top_k)
    else:
        st.warning("⚠ FAISS index is empty. Upload a valid dataset.")
        return ""

    retrieved_hashtags = set()
    for idx in indices[0]:
        if 0 <= idx < len(df):
            raw_hashtags = df.iloc[idx]['hashtags']
            retrieved_hashtags.update(f"#{tag.strip().lstrip('#')}" for tag in raw_hashtags.split(","))

    if not retrieved_hashtags:
        st.warning("⚠ No similar posts found. Try adding more data.")
        return ""

    # Use GPT-4 to generate additional hashtags
    gpt_prompt = f"""
    The following hashtags were retrieved from similar posts: {', '.join(retrieved_hashtags)}
    Suggest 5 additional relevant and trending hashtags.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in social media growth strategies."},
            {"role": "user", "content": gpt_prompt}
        ]
    )

    gpt_response = response.choices[0].message.content.strip()
    gpt_suggestions = [f"#{re.sub(r'^\d+\.\s*', '', line).strip().lstrip('#')}" for line in gpt_response.split("\n") if line.strip()]

    final_hashtags = list(retrieved_hashtags) + gpt_suggestions
    return ", ".join(final_hashtags)

# --- Streamlit UI ---
user_input = st.text_area("📝 Enter your post:")

if st.button("🔎 Generate Hashtags"):
    if user_input.strip():
        hashtags = recommend_hashtags(user_input)
        st.subheader("🔹 Suggested Hashtags:")
        st.write(hashtags)
    else:
        st.warning("⚠ Please enter a post before generating hashtags.")
