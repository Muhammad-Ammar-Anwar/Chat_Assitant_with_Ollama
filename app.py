import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Function to load Lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a Lottie animation (optional)
lottie_ai = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Gemma Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 LangChain Chat with Gemma 2B")
st.markdown("Ask me anything! Powered by **LangChain** and **Ollama's Gemma 2B** model.")

# Display animation
with st.container():
    st_lottie(lottie_ai, height=200, key="ai")

# Input field
st.markdown("### 💬 What question do you have in mind?")
input_text = st.text_input("Type your question here...")

# LangChain setup
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question:{question}")
    ]
)

llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Display result
if input_text:
    with st.spinner("Thinking... 💭"):
        try:
            response = chain.invoke({"question": input_text})
            st.success("✅ Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
