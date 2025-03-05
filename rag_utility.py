from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import os 
import json 
import re 



working_dir = os.path.dirname(os.path.abspath((__file__)))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# modelo de hugging face para volver un espacio vectorial la data de conocimiento
model_name = "sentence-transformers/all-MiniLM-L6-v2"
#model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding = HuggingFaceEmbeddings()

# api por la cual consumimos el modelo llm (Deepseek)
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0
)

# Definir el prompt 
template = """Responde siempre en **español** con un tono **profesional, técnico y objetivo**, siguiendo estos pasos de manera estricta:

1. **Lectura y análisis del contexto**  
   - Revisa detenidamente todo el contexto.  
   - No uses información externa ni hagas suposiciones.  
   - Si detectas datos contradictorios o ambiguos, señalálos y decide si aún es posible responder con la información proporcionada.

2. **Identificación de datos relevantes**  
   - Extrae únicamente los elementos que tengan relación directa con la pregunta.  
   - Desecha todo lo que no contribuya a la respuesta.

3. **Determinación de la suficiencia de la información**  
   - Si el contexto **no** brinda información suficiente o resulta **incongruente**, responde:  
     `"No hay información suficiente en el documento."`  
   - De lo contrario, continúa con la siguiente fase.

4. **Elaboración de la respuesta**  
   a. **Breve introducción** (1-2 líneas) indicando la disponibilidad de información.  
   b. **Desarrollo** en **formato numerado**, referenciando de forma explícita los datos clave del contexto (por ejemplo: “Según el contexto se menciona que…”)  
   c. **Conclusión** con 1-2 líneas que unan los puntos esenciales y confirmen la respuesta final.  

5. **Revisión final**  
   - Verifica la precisión de tu respuesta y que no existan contradicciones.  
   - Asegúrate de **no incluir información adicional o especulativa** que no aparezca en el contexto.  
   - Mantén la **extensión de la respuesta concisa** y enfocada en lo solicitado.
   - Da detalle de la informacion, como explicacion de cada concepto.
   - Dar informacion de categorias subcategorias

---

### **Contexto:**  
{context}

### **Pregunta:**  
{question}

### **Respuesta:**"""

# Crear el objeto de prompt template
prompt_template = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# def process_document_to_chroma_db(file_name):
#     """Carga y procesa un documento PDF en la base de datos Chroma."""
#     loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
#     documents = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=2000,
#         chunk_overlap=200
#     )
#     texts = text_splitter.split_documents(documents)

#     vectordb = Chroma.from_documents(
#         documents=texts,
#         embedding=embedding,
#         persist_directory=f"{working_dir}/doc_vectorstore"
#     )
#     return 0
def process_document_to_faiss(file_name):
    """Carga y procesa un documento PDF en una base de datos FAISS."""
    formato = file_name.split('.')[-1]
    if formato == 'pdf':
        loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Crear la base de datos FAISS
        vectordb = FAISS.from_documents(
            documents=texts,
            embedding=embedding
        )
    elif formato == 'json':
        # Cargar JSON manualmente
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)

        documentos = []
        for item in data:
            # Limpiar títulos eliminando etiquetas HTML
            titles = " ".join([clean_html(t) for t in item.get("titles", [])])

            # Limpiar párrafos filtrando contenido vacío o irrelevante
            paragraphs = " ".join([p for p in item.get("paragraphs", []) if clean_paragraph(p)])
            
            # Concatenar títulos y párrafos en un solo texto
            full_text = f"{titles}\n\n{paragraphs}".strip()
            
            if full_text:  # Evitar documentos vacíos
                documentos.append(Document(page_content=full_text)) 
        # Dividir documentos en fragmentos pequeños para optimizar la indexación
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documentos)


        # Guardar en FAISS
        vectordb = FAISS.from_documents(
            documents=texts,
            embedding=embedding  
        ) 

    # Guardar la base de datos FAISS para uso futuro
    vectordb.save_local(f"{working_dir}/faiss_vectorstore")
    return 0

def clean_html(text):
    """Elimina etiquetas HTML y espacios extra de un texto."""
    clean_text = re.sub(r"<.*?>", "", text)  # Elimina etiquetas HTML
    return clean_text.strip()

def clean_paragraph(text):
    """Filtra contenido vacío o irrelevante."""
    exclude_phrases = { "", "\n"}
    return text.strip() not in exclude_phrases

# def answer_question(user_question):
#     """Genera una respuesta a una pregunta basándose en los documentos indexados en Chroma."""
#     vectordb = Chroma(
#         persist_directory=f"{working_dir}/doc_vectorstore",
#         embedding_function=embedding
#     )
#     retriever = vectordb.as_retriever()

#     # Crear la cadena de respuesta con el template de prompt
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={"prompt": prompt_template}  # Integración del template
#     )

#     response = qa_chain.invoke({"query": user_question})
#     answer = response["result"]
def answer_question(user_question):
    """Genera una respuesta a una pregunta basándose en los documentos indexados en FAISS."""
    # Cargar la base de datos FAISS
    vectordb = FAISS.load_local(
        folder_path=f"{working_dir}/faiss_vectorstore",
        embeddings=embedding,
        allow_dangerous_deserialization=True  # Necesario para cargar desde disco
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # Recuperar los 5 documentos más relevantes

    # Crear la cadena de respuesta con el template de prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer


