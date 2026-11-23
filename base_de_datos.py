from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from schema import DatosContrato, Parte

from prompts import EXTRACCION_CONTRATO_PROMPT

from config import (
    CONTRATOS_PATH,
    CHROMA_DB_PATH,
    CHUNKS_PATH,
    EXTRACCION_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EXTRACTION_MODEL
)

import pickle
import shutil
import json
import os

if os.path.exists(CHROMA_DB_PATH):
    shutil.rmtree(CHROMA_DB_PATH)
    print("Base anterior eliminada")

llm_extractor = ChatOpenAI(model=EXTRACTION_MODEL, temperature=0)
extractor = PromptTemplate.from_template(EXTRACCION_CONTRATO_PROMPT) | llm_extractor.with_structured_output(DatosContrato)

#funcion cabecera
def formatear_parte(parte: Parte) -> str:
    if parte.representante:
        return f"{parte.nombre} (rep. {parte.representante})"
    return parte.nombre

def construir_cabecera(datos: DatosContrato) -> str:
    return (
        f"[{datos.tipo_contrato} | {datos.ciudad} | "
        f"Arrendador: {formatear_parte(datos.arrendador)} | "
        f"Arrendatario: {formatear_parte(datos.arrendatario)} | "
        f"Inmueble: {datos.direccion_inmueble}]"
    )

archivos = [f for f in os.listdir(CONTRATOS_PATH) if f.endswith(".pdf")]
print(f"Encontrados {len(archivos)} contratos")

todos_los_chunks = []
tabla_contratos = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)

for archivo in archivos:
    ruta = os.path.join(CONTRATOS_PATH, archivo)
    print(f"Procesando {archivo}")

    paginas = PyPDFLoader(ruta).load()
    texto_completo = "\n".join(p.page_content for p in paginas)

    datos = extractor.invoke({"text":texto_completo})
    cabecera = construir_cabecera(datos)
    print(cabecera)

    chunks = text_splitter.split_documents(paginas)
    for c in chunks:
        c.page_content = f"{cabecera}\n{c.page_content}"

    todos_los_chunks.extend(chunks)

    registro = datos.model_dump()
    registro["archivo"] = archivo
    tabla_contratos.append(registro)

    print(f"{len(chunks)} chunks")


with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(todos_los_chunks, f)

vectorstore = Chroma.from_documents(
    todos_los_chunks,
    embedding=embedding,
    persist_directory=CHROMA_DB_PATH
)

with open(EXTRACCION_PATH, "w", encoding="utf-8") as f:
    json.dump(tabla_contratos, f, ensure_ascii=False, indent=2)

for i, c in enumerate(todos_los_chunks[:3]):
    print(f"\n{'='*60}")
    print(f"CHUNK {i} — {len(c.page_content)} chars")
    print(f"{'='*60}")
    print(c.page_content[:300])


print(f"\n{'='*60}")

print(f"Chunks en pickle:     {len(todos_los_chunks)}")
print(f"Documentos en Chroma: {vectorstore._collection.count()}")