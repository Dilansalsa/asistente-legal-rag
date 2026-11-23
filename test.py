# # test_chain.py
# from rag_system import inicialize_rag_system

# rag_chain, retriever = inicialize_rag_system()

# pregunta = "¿qué contratos hay en Valencia?"

# print(f"TIPO DE CADENA: {type(rag_chain)}")

# r = rag_chain.invoke(pregunta)

# print("=" * 60)
# print(f"TIPO: {type(r)}")
# print(f"CLAVES: {list(r.keys())}")
# print("=" * 60)

# print(f"\nQUESTION: {r['question']}")

# print(f"\nCONTEXT — {len(r['context'])} documentos:")
# for i, doc in enumerate(r['context'], 1):
#     fuente = doc.metadata.get('source', '?').split("\\")[-1]
#     pagina = doc.metadata.get('page', '?')
#     print(f"\n  [{i}] {fuente} — pág {pagina} — {len(doc.page_content)} chars")
#     print(f"      {doc.page_content[:200]}...")

# print(f"\n{'=' * 60}")
# print(f"RESPUESTA:\n{r['respuesta']}")

# print("="*60)

# #print(rag_chain)

# print(len(r["context"]))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from schema import RutaConsulta
from prompts import CLASIFICADOR_PROMPT

llm_clasificador = ChatOpenAI(model="gpt-4o-mini", temperature=0)
clasificador = PromptTemplate.from_template(CLASIFICADOR_PROMPT) | llm_clasificador.with_structured_output(RutaConsulta)

preguntas = [
    "cuales son las personas involucradas en los arrendamientos de negocios",
    "¿cuál es el importe exacto de la garantía bancaria del local de Calle Colón?",
    "¿en qué contratos ambas partes son personas jurídicas?",
    "¿qué contratos prohíben la tenencia de mascotas?",
    "¿cuál es la penalización por desistimiento en Sevilla comparada con Madrid?",
]

for p in preguntas:
    r = clasificador.invoke({"question": p})
    print(f"{r.ruta} | {p}")
    print(f"           └─ {r.razon}\n")