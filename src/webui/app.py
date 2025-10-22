import streamlit as st
import requests

st.title('ChatLegal — Chatbot Jurídico')
question = st.text_input('Digite sua pergunta jurídica')
if st.button('Enviar') and question:
    with st.spinner('Consultando o modelo...'):
        resp = requests.post('http://localhost:8000/chat', json={'question': question, 'top_k': 8})
        if resp.status_code == 200:
            data = resp.json()
            st.subheader('Resposta')
            st.write(data.get('answer'))
            st.subheader('Fontes recuperadas')
            for s in data.get('sources', []):
                st.write('- ' + s)
        else:
            st.error('Erro na API')
