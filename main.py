# app.py
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator  # 👈 Nuevo

# ----------------------------
# 1️⃣ Crear/actualizar embeddings
# ----------------------------
def actualiza_embedding(pdf_path: str, persist_dir: str):
    st.info("📄 Cargando PDF y creando embeddings...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        chunked_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    st.success("✅ Embeddings creados y guardados en ./chroma_db")
    return vectordb


# ----------------------------
# 2️⃣ Interfaz Streamlit
# ----------------------------
st.set_page_config(page_title="Mesa de Ayuda IA (Local)", page_icon="🤖")
st.title("🤖 Mesa de Ayuda IA (Local)")
st.write("Agente de asistencia técnica que responde con base en el contenido del documento cargado.")

pdf_path = "data/manual.pdf"
persist_dir = "./chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if os.path.exists(persist_dir):
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    vectordb = None
    st.warning("⚠️ No hay base de datos vectorial. Debes actualizar embeddings primero.")

if st.button("🔄 Actualizar Embeddings"):
    if os.path.exists(pdf_path):
        vectordb = actualiza_embedding(pdf_path, persist_dir)
    else:
        st.error("❌ No se encontró el PDF en data/manual.pdf")


# ----------------------------
# 3️⃣ Configurar LLM local desde LM Studio
# ----------------------------
llm = ChatOpenAI(
    model="mistral-7b-instruct-v0.2.Q4_K_M",
    openai_api_base="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.1,
    max_tokens=1200
)

prompt_template = """
Eres un agente de mesa de ayuda especializado en software empresarial.
Tu tarea es responder SIEMPRE en ESPAÑOL, sin excepciones.
Debes redactar respuestas claras, extensas y con explicaciones técnicas.
Usa ÚNICAMENTE la información del manual proporcionado.
Si la información no está en el contexto, responde exactamente:
"No encuentro esa información en el documento."

Pregunta del usuario: {input}
Contexto extraído del manual: {context}
"""

qa_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)


# ----------------------------
# 4️⃣ Búsqueda y respuesta con barra de estado
# ----------------------------
pregunta = st.text_area("✍️ Escribe tu pregunta aquí:")

if st.button("📨 Enviar"):
    if not vectordb:
        st.error("⚠️ Primero genera embeddings con el botón '🔄 Actualizar Embeddings'.")
    elif pregunta:
        progress_bar = st.progress(0, text="⏳ Procesando tu pregunta...")

        progress_bar.progress(25, text="🔍 Buscando información relevante en el manual...")
        resultados = vectordb.similarity_search(pregunta, k=8)
        contexto = " ".join([doc.page_content for doc in resultados])

        progress_bar.progress(60, text="⚙️ Generando respuesta detallada con el modelo...")
        respuesta = qa_chain.invoke({"input": pregunta, "context": contexto})

        # 👇 Traducir siempre la salida al español
        respuesta_final = GoogleTranslator(source="auto", target="es").translate(respuesta["text"])

        progress_bar.progress(100, text="✅ Respuesta lista")

        st.subheader("📘 Respuesta:")
        st.write(respuesta_final)
    else:
        st.warning("⚠️ Por favor ingresa una pregunta antes de enviar.")
