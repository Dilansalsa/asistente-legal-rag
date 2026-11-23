# Asistente Legal con RAG

Asistente que responde preguntas sobre contratos de arrendamiento en PDF, combinando **extracción estructurada de datos** con **búsqueda híbrida**.

## ¿Cómo funciona?

**1. Procesamiento de contratos (`base_de_datos.py`)**
- Lee cada contrato en PDF.
- Un LLM extrae los datos clave a un formato estructurado (Pydantic → JSON): tipo de contrato, ciudad, arrendador, arrendatario (con representante si aplica) y dirección del inmueble.
- Divide el texto en fragmentos (chunks) y añade a cada uno una cabecera con los datos de su contrato, para que la búsqueda sepa a qué contrato pertenece cada fragmento.
- Guarda los fragmentos en ChromaDB.

**2. Consulta (`rag_system.py` + `app.py`)**
- **Multi-Query:** un LLM reformula la pregunta en varias versiones para encontrar más información relevante.
- **Búsqueda híbrida:** combina búsqueda semántica con MMR (resultados relevantes y variados) y BM25 (palabras clave exactas).
- Genera la respuesta con los fragmentos encontrados, indicando fuente y página.
- La interfaz incluye un **monitor de tokens y costo** de cada consulta.

## Tecnologías

Python · LangChain · OpenAI API (GPT-4o-mini, text-embedding-3-small) · ChromaDB · BM25 · Pydantic · Streamlit

## Instalación

```bash
pip install streamlit langchain-openai langchain-chroma langchain-community langchain-classic langchain-text-splitters pypdf rank_bm25
```

Configura tu API key de OpenAI como variable de entorno:

```bash
# PowerShell
$env:OPENAI_API_KEY="tu-api-key"
# Linux / Mac
export OPENAI_API_KEY="tu-api-key"
```

## Uso

1. Coloca los contratos en PDF en una carpeta `contratos/` **al mismo nivel** que la carpeta del proyecto (ver rutas en `config.py`).
2. Procesa los contratos (una sola vez):
   ```bash
   python base_de_datos.py
   ```
3. Inicia la app:
   ```bash
   streamlit run app.py
   ```
