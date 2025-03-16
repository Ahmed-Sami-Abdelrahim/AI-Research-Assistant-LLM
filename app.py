import streamlit as st
from answer_generator import generate_answer

st.title("AI Research Assistant")
st.write("Ask a research question and get an AI-generated answer based on ArXiv papers.")

question = st.text_input("Enter your research question:")

if st.button("Get Answer"):
    if question:
        answer = generate_answer(question)
        st.write("**Answer:**")
        st.write(answer)
    else:
        st.write("Please enter a question.")
