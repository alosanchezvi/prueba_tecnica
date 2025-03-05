# prueba_tecnica

Se implementará un RAG (Retrieval-Augmented Generation), donde el orquestador será LangChain. Se utilizarán embeddings de Hugging Face y el modelo DeepSeek 70B de Groq a través de su API, con el objetivo de proporcionar respuestas a las preguntas de TUYA.

La base de información se obtendrá mediante web scraping de las páginas web de TUYA, asegurando que el sistema tenga acceso a datos actualizados y relevantes.


# Desarrollo 
Se utiliza Scrapy para extraer datos de las páginas de Tuya.
El código en spider.py obtiene el contenido relevante de la web.
Los datos extraídos se almacenan en output_v1.json para su posterior procesamiento.


Se carga los datos de output_v1.json.
Divide la información en trozos más pequeños para facilitar su uso en modelos de IA.
Se usa un modelo de Hugging Face para convertir los fragmentos de texto en embeddings vectoriales.
Estos embeddings se almacenan en un índice de FAISS para permitir una búsqueda eficiente.


Consulta de Preguntas con Similaridad de FAISS
Cuando un usuario hace una consulta, el sistema:
Convierte la pregunta en un embedding usando el mismo modelo de Hugging Face.
Busca en FAISS los fragmentos más similares a la consulta.
Selecciona los embeddings más relevantes para contextualizar la respuesta.
Una vez que se han recuperado los fragmentos relevantes, se pasan al modelo DeepSeek.
DeepSeek genera una respuesta robusta basada en el contexto proporcionado


# implementacion 

 se crea un ambiente virtual se intalan las dependencia de requiremensts.txt 
 
 se despliega con el comando *streamlit run main.py*
