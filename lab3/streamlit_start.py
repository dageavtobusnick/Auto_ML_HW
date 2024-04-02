import streamlit as st
import main
prompt = st.chat_input("Введите фразу для перевода:")
if prompt:
    st.write(main.translate_phrase(prompt))