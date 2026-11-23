import os

#Rutas
BASE = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(BASE, "..", "fuente_de_datos_externa")

CONTRATOS_PATH = os.path.join(BASE, "..", "contratos")
CHROMA_DB_PATH = os.path.join(DATOS, "Base_vectorial_chroma")
CHUNKS_PATH = os.path.join(DATOS, "chunks.pkl")
EXTRACCION_PATH = os.path.join(DATOS, "contratos_extraidos.json")

#Configuracion modelos
EMBEDDING_MODEL = "text-embedding-3-small"
QUERY_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o-mini"
EXTRACTION_MODEL = "gpt-4o-mini"

#Configuracion del chunking
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 250

#Configuracion del retriever
SEARCH_TYPE = "mmr"
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20
SEARCH_K = 4

#Configuracion retriever hibrido
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.75

#BM25
BM25_K = 3
ENSEMBLE_WEIGHTS = [0.6, 0.4]