import os
import json

import streamlit as st

from rag_utility import process_document_to_faiss, answer_question


# Directorio actual
working_dir = os.getcwd()

st.title("Asistente Virtual RAG(DeepSeek)")

# file uploader widget
uploaded_file = st.file_uploader("Sube un archivo pdf o json", type=["pdf", "json"])

if uploaded_file is not None:
    # Definicion de direccion
    save_path = os.path.join(working_dir, uploaded_file.name)
    
    # salvar el archivo
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_extension = uploaded_file.name.split(".")[-1]

    # Recuperacion de la base de conocimiento
    process_document = process_document_to_faiss(uploaded_file.name)
    st.info("Documento procesado exitosamente")
    
#  input de la consulta
user_question = st.text_area("Has tu pregunta sobre el documento")

if st.button("Respuesta"):

    answer = answer_question(user_question)

    st.markdown("### DeepSeek-R1 Response")
    st.markdown(answer)