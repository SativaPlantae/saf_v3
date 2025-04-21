import os
import streamlit as st
import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Carrega chave da OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# Função para carregar e preparar os dados
def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv, sep=";")
    texto_unico = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))
    return Document(page_content=texto_unico)

# Carrega a cadeia com memória de conversa
@st.cache_resource
def carregar_chain():
    documento = carregar_dados("data.csv")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents([documento])

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
Você é um assistente virtual treinado com base em dados do Sistema Agroflorestal SAF Cristal.
Fale de forma simples e direta. Se não souber algo, diga isso naturalmente.

Histórico:
{chat_history}

Informações relevantes:
{context}

Pergunta: {question}
Resposta:"""
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Interface do Streamlit
st.set_page_config(page_title="Chatbot SAF Cristal", page_icon="🌱")
st.title("🌱 Chatbot do SAF Cristal")

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = carregar_chain()

# Mostrar histórico
for remetente, mensagem in st.session_state.mensagens:
    with st.chat_message("user" if remetente == "🧑‍🌾" else "assistant", avatar=remetente):
        st.markdown(mensagem)

user_input = st.chat_input("Digite sua pergunta aqui...")

if user_input:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_input)
    st.session_state.mensagens.append(("🧑‍🌾", user_input))

    with st.spinner("Consultando o SAF Cristal..."):
        try:
            resposta = st.session_state.qa_chain.run(user_input)
        except Exception as e:
            resposta = f"⚠️ Erro: {e}"

    with st.chat_message("assistant", avatar="🐝"):
        st.markdown(resposta)
    st.session_state.mensagens.append(("🐝", resposta))
