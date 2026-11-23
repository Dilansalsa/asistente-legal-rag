from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from schema import DatosContrato
from prompts import EXTRACCION_CONTRATO_PROMPT

loader = PyPDFLoader(r"C:\Users\Dilan\Documents\curso_langchain\Tema 3\contratos\CONTRATO DE ARRENDAMIENTO DE LOCAL DE NEGOCIO.pdf")
paginas = loader.load()
texto = "\n".join(p.page_content for p in paginas)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
extractor = PromptTemplate.from_template(EXTRACCION_CONTRATO_PROMPT) | llm.with_structured_output(DatosContrato)

datos = extractor.invoke({"text": texto})

print(datos)
print()
print(datos.model_dump_json(indent=2))