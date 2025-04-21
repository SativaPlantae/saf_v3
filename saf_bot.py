import os
import streamlit as st
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# 🔐 Chave da OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def carregar_chain_com_memoria():
    df = pd.read_csv("data.csv", sep=";")

    # Junta tudo em um único texto
    texto_unico = "\n".join(df.astype(str).apply(lambda x: " | ".join(x), axis=1))
    document = Document(page_content=texto_unico)

    # Divide em pedaços menores
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents([document])

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""
Você é um assistente treinado com base em dados do SAF Cristal.
Seja claro, direto e acessível. Use o histórico da conversa. 
Se não souber, diga isso com naturalidade.

Histórico:
{chat_history}

Informações:
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

# Interface
st.set_page_config(page_title="Chatbot SAF Cristal 🌱", page_icon="🐝")
st.title("🐝 Chatbot do SAF Cristal")
st.markdown("Converse com o assistente sobre o Sistema Agroflorestal Cristal 📊")

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = carregar_chain_com_memoria()

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
