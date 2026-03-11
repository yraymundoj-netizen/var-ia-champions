import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.rag.rag_chat import ask_question

st.title("⚽ VAR-IA Chat")

question = st.text_input("Haz una pregunta")

if question:
    with st.spinner("Analizando jugada..."):
        response = ask_question(question)

    st.write(response)