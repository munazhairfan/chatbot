import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os
HF_API_KEY = os.getenv("HF_API_KEY")

st.set_page_config(page_title="Unit Converter", page_icon="🌷")

# Model Endpoints
TEXT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Handles multiple tasks
IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

def query_huggingface(model, payload):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"error": "Invalid response from Hugging Face API"}


def generate_text_response(user_input):
    payload = {
        "inputs": user_input,
        "parameters": {
            "max_new_tokens": 200,  # Limits response length
            "temperature": 0.3,  # Reduces randomness
            "top_p": 0.9,  # Controls diversity
        }
    }
    response = query_huggingface(TEXT_MODEL, payload)

    if isinstance(response, list) and "generated_text" in response[0]:
        return response[0]["generated_text"]
    else:
        return "Sorry, I couldn't process that."

def generate_image(prompt):
    API_URL = f"https://api-inference.huggingface.co/models/{IMAGE_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return None, f"Error: {response.status_code} - {response.text}"

    try:
        image = Image.open(BytesIO(response.content))
        return image, None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


# Function to summarize text
def summarize_text(text):
    payload = {"inputs": text}
    response = query_huggingface(SUMMARIZATION_MODEL, payload)

    if isinstance(response, list) and "summary_text" in response[0]:
        return response[0]["summary_text"]
    else:
        return "Sorry, I couldn't summarize that. Please try again."

# Function to analyze sentiment
def analyze_sentiment(text):
    payload = {"inputs": text}
    response = query_huggingface(SENTIMENT_MODEL, payload)

    if isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
        sentiment_data = response[0]  
 
        best_sentiment = max(sentiment_data, key=lambda x: x["score"])
        
        return f"Sentiment: {best_sentiment['label']} (Confidence: {best_sentiment['score']:.2f})"

    return "Error: Unexpected API response format"

# Streamlit UI
with st.sidebar:
    st.header("🌷 About Peppy")
    st.markdown("""
    **Peppy** is a multi-functional chatbot that can:
    - 📝 Summarize text
    - 😊 Analyze sentiment
    - 🎨 Generate images
    - 💬 Engage in conversations  
      
    Try it out by entering a prompt! 🚀
    """)

st.markdown("""
    <style>
        /* Background Styling */
       .stApp {
            font-family: 'Arial', sans-serif;
            background-color: #fafdf7;
            background-image:  
                linear-gradient(rgba(233, 69, 247, 0.4) 0.8px, transparent 0.8px), 
                linear-gradient(to right, rgba(233, 69, 247, 0.4) 0.8px, #fafdf7 0.8px);
            background-size: 16px 16px;
        }

        /* Button Styling */
        .stButton > button {
            border-radius: 8px;
            background-color: #4CAF50 !important;
            color: white !important;
            padding: 8px 12px;
            font-size: 16px;
            border: none;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #45a049 !important;
            transform: scale(1.05);
        }
        .stButton > button:active {
            background-color: #3e8e41 !important;
            color: white !important;
        }

        /* Input Field Styling */
        .stTextInput > div > div > input {
            border: 2px solid #ccc !important;
            border-radius: 5px;
            padding: 8px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #e2619f !important;
            border: 10px;
            outline: none !important;
        }

        /* Chat Message Styling */
        .chat-container {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
    </style>

    <h1 style='text-align: center; color: #4CAF50;'>🌷 Welcome to <span style='color: #e2619f;'>Peppy</span>!</h1>
    <p style='text-align: center; font-size:18px;'>Your AI-powered multi-tasking chatbot!</p>
""", unsafe_allow_html=True)



user_input = st.text_input("Ask me anything!")

if st.button("Submit"):
    with st.spinner("⏳ Generating response..."):
        if "generate" in user_input.lower():
            image, error = generate_image(user_input)
            if error:
                st.error(error)
            else:
                st.image(image, caption="Generated Image")

        elif "summarize" in user_input.lower():
            summary = summarize_text(user_input.replace("summarize", "").strip())
            st.write(f"**Summary:** {summary}")

        elif "sentiment" in user_input.lower():
            sentiment = analyze_sentiment(user_input.replace("analyze sentiment of", "").strip())
            st.write(f"**Sentiment:** {sentiment}")

        else:
            response = generate_text_response(user_input)
            st.write(f"**Peppy:** {response}")
