from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from langchain_community.callbacks import get_openai_callback
import streamlit as st
import pickle
import json

from config import *
from prompts import *

def parte_tabla(parte: dict) -> str:
    if parte.get("representante"):
        return f"{parte['nombre']} (rep. {parte['representante']})"
    return parte["nombre"]

def cargar_tabla() -> str:
    with open(EXTRACCION_PATH, "r", encoding="utf-8") as f:
        contratos = json.load(f)

    lineas = []
    for i, c in enumerate(contratos, 1):
        lineas.append(
            f"{i}. {c['tipo_contrato']} | {c['ciudad']} | "
            f"Arrendador: {parte_tabla(c['arrendador'])} | "
            f"Arrendatario: {parte_tabla(c['arrendatario'])} | "
            f"Inmueble: {c['direccion_inmueble']}"
        )
    return "\n".join(lineas)

@st.cache_resource
def inicialize_rag_system():

    tabla_texto = cargar_tabla()
    
    #vectore store
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DB_PATH
    )

    #Modelos
    llm_queries = ChatOpenAI(model=QUERY_MODEL, temperature=0)
    llm_generation = ChatOpenAI(model=GENERATION_MODEL, temperature=0)

    #retriever
    base_retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k":SEARCH_K, 
            "lambda_mult":MMR_DIVERSITY_LAMBDA,
            "fetch_k":MMR_FETCH_K
        }
    )

    #retriever para comparacion
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k":SEARCH_K
        }
    )

    #prompt para multyquery
    multi_query_prompt=PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    #multiquery retriever com prompt
    mmr_multi_retriever =MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm_queries, 
        prompt=multi_query_prompt
    )

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = BM25_K

    #ensamble retriever MMR y similarity
    if ENABLE_HYBRID_SEARCH:
        final_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, bm25_retriever],
            weights=ENSEMBLE_WEIGHTS
        )
    else:
        final_retriever = mmr_multi_retriever

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    #Preprocesar documentos separados
    def format_docs(docs):
        formatted = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"

            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Pagina: {doc.metadata['page']}"

            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")

        return "\n\n".join(formatted)

    rag_chain =(
        RunnableParallel({
            "context":final_retriever,
            "question":RunnablePassthrough()
        }).assign(
            respuesta=(
                RunnableLambda(lambda x: {
                        "context": format_docs(x["context"]),
                        "question": x["question"]
                })
                | prompt
                | llm_generation
                | StrOutputParser()
            )
        ) 
    )

    return rag_chain, final_retriever

def query_rag(question):
    try:
        rag_chain, retriever = inicialize_rag_system()

        with get_openai_callback() as cb:
            resultado = rag_chain.invoke(question)

        response = resultado["respuesta"] 
        docs = resultado["context"]

        costo_info = {
            "tokens": cb.total_tokens,
            "tokens_entrada": cb.prompt_tokens,
            "tokens_salida": cb.completion_tokens,
            "costo": cb.total_cost,
            "llamadas": cb.successful_requests
        }

        docs_info = []
        for i, doc in enumerate(docs, 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content,
                "fuente": doc.metadata.get('source', 'No especificada').split("\\")[-1],
                "pagina": doc.metadata.get('page', 'No especificada')
            }
            docs_info.append(doc_info)

        return response, docs_info, costo_info
    except Exception as e:
        costo_vacio = {"tokens": 0, "tokens_entrada": 0, "tokens_salida": 0, "costo": 0.0, "llamadas": 0}
        return f"Error al procesar la consulta {str(e)}", [], costo_vacio

def get_retriever_info():
    return {
        "tipo":f"{SEARCH_TYPE.upper()} + Multiquery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos":SEARCH_K,
        "diversidad":MMR_DIVERSITY_LAMBDA,
        "candidatos":MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }