import streamlit as st
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from dotenv import load_dotenv
import os

# ====================== TESSERACT PATH ======================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# =========================================================

load_dotenv()

# Cache Model
@st.cache_resource
def initialize_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = initialize_embedding_model()

def get_db_connection():
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        register_vector(conn)
        return conn
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        st.info("Please check your DATABASE_URL in .env file")
        return None

def setup_openai_client():
    api_key = os.getenv("API_KEY")
    if not api_key or api_key.startswith("your_"):
        st.error("❌ OpenRouter API Key is missing. Please add it in .env file")
        return None
    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        return client
    except Exception as e:
        st.error(f"OpenAI Client Error: {e}")
        return None

openai_client = setup_openai_client()

def split_into_chunks(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ================== Beautiful Title ==================
st.markdown("""
    <h1 style='text-align: center; 
               background: linear-gradient(90deg, #ff3366, #00ffff, #ffcc00);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-size: 3.5rem;
               font-weight: bold;
               margin-bottom: 0.2rem;'>
        VISUAL RAG
    </h1>
    <h3 style='text-align: center; color: #00ffcc; margin-top: 0;'>
        Chakiri Shanmukha Sai
    </h3>
""", unsafe_allow_html=True)

st.caption("📍 KL University, Vijayawada | AI-Powered Visual Intelligence")

uploaded_files = st.file_uploader("Upload Document Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Process Images"):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    elif not openai_client:
        st.error("Cannot process without valid API Key.")
    else:
        try:
            full_text = ""
            for file in uploaded_files:
                image = Image.open(file)
                text = pytesseract.image_to_string(image)
                full_text += text + "\n"

            if not full_text.strip():
                st.warning("No text could be extracted from images.")
            else:
                chunks = split_into_chunks(full_text)
                embeddings = embedding_model.encode(chunks)
                conn = get_db_connection()
                if conn:
                    cur = conn.cursor()
                    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")
                    for chunk, emb in zip(chunks, embeddings):
                        cur.execute("INSERT INTO documents (chunk_text, embedding) VALUES (%s, %s)", (chunk, emb))
                    conn.commit()
                    cur.close()
                    conn.close()
                    st.success(f"✅ {len(uploaded_files)} Images processed successfully!")
                    st.session_state.processed = True
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

# Question Section
if st.session_state.get("processed", False):
    question = st.text_input("Ask a question about the uploaded images:")
    if st.button("Get Answer"):
        if question.strip() and openai_client:
            try:
                with st.spinner("Thinking..."):
                    query_vector = embedding_model.encode([question])[0]
                    conn = get_db_connection()
                    if conn:
                        cur = conn.cursor()
                        cur.execute("SELECT chunk_text FROM documents ORDER BY embedding <=> %s LIMIT 3", (query_vector,))
                        results = cur.fetchall()
                        context = "\n".join([row[0] for row in results])
                        cur.close()
                        conn.close()
                        prompt = f"""Answer based on the context only:\n\nContext: {context}\n\nQuestion: {question}"""
                       
                        response = openai_client.chat.completions.create(
                            model="openai/gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.success(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #ff99cc;'>Made with ❤️ by Chakiri Shanmukha Sai</p>", unsafe_allow_html=True)