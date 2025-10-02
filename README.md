# 🤖 Mesa de Ayuda IA (Local)

Este proyecto implementa una **mesa de ayuda inteligente** que responde preguntas basadas en un manual en PDF, utilizando **modelos LLM locales** con LM Studio y LangChain.

<img width="763" height="826" alt="image" src="https://github.com/user-attachments/assets/63857e4d-a349-4089-95ba-c1d750a7a220" /> <img width="724" height="724" alt="image" src="https://github.com/user-attachments/assets/fa78eef6-5aa9-4445-9ec2-637f01ef0e67" />

---

## 🚀 Características

* Carga manuales en PDF y genera embeddings para construir una base de conocimiento vectorial.
* Permite hacer preguntas en **lenguaje natural** y obtiene respuestas claras y técnicas.
* Responde siempre en **español**, incluso si el modelo genera texto en otro idioma.
* Corre **100% en local**, sin necesidad de servicios en la nube.
* Interfaz sencilla construida con **Streamlit**.

---

## 🛠️ Tecnologías utilizadas

* **Python 3.10+**
* **Streamlit** (interfaz de usuario)
* **LangChain** (orquestación de prompts y contexto)
* **ChromaDB** (almacenamiento vectorial)
* **HuggingFace Embeddings**
* **LM Studio** con el modelo **Mistral-7B Instruct**

<img width="1365" height="718" alt="image" src="https://github.com/user-attachments/assets/6f676fe6-e70c-4e7f-85e3-393ed29446af" />

---

## 📂 Estructura del proyecto

```
📁 proyecto-mesa-ayuda-ia
 ├── data/              # Manuales PDF
 ├── chroma_db/         # Base de datos vectorial persistida
 ├── main.py            # Código principal Streamlit
 ├── requerimientos.txt # Dependencias
```

---

## ▶️ Cómo ejecutar

1. Instala las dependencias:

   ```bash
   pip install -r requerimientos.txt
   ```
2. Abre **LM Studio**, carga el modelo `mistral-7b-instruct-v0.2.Q4_K_M` y actívalo en `http://localhost:1234`.
3. Ejecuta la app:

   ```bash
   streamlit run main.py
   ```
4. Abre el navegador en `http://localhost:8501` y sube un manual PDF para generar embeddings.

---

## 📽️ Demo

El sistema permite hacer preguntas como:
❓ *“¿Cómo modificar una solicitud de pedidos en SAP?”*
✅ Y devuelve una explicación técnica detallada basada en el manual.

---

## 📌 Futuras mejoras

* Soporte para múltiples manuales en paralelo.
* Citado de páginas específicas del documento.
* Interfaz multilingüe sin necesidad de traductor externo.
