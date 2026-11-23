import streamlit as st
from rag_system import query_rag, get_retriever_info
from config import QUERY_MODEL, GENERATION_MODEL

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG - Asistente Legal",
    page_icon="⚖️",
    layout="wide"
)

# Título
st.title("⚖️ Sistema RAG - Asistente Legal")
st.divider()

# Inicializar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Inicializar el acumulador de costos más detallado
if "costo_total" not in st.session_state:
    st.session_state.costo_total = {
        "tokens": 0, 
        "tokens_entrada": 0,
        "tokens_salida": 0,
        "costo": 0.0, 
        "llamadas": 0,
        "consultas": 0
    }

# Sidebar completo con monitor de gastos
with st.sidebar:
    st.header("📋 Información del Sistema")
    
    # Información del retriever
    retriever_info = get_retriever_info()
    
    st.markdown("**🔍 Retriever:**")
    st.info(f"Tipo: {retriever_info['tipo']}")
    
    st.markdown("**🤖 Modelos:**")
    st.info(f"Consultas: {QUERY_MODEL}\n\nRespuestas: {GENERATION_MODEL}")
    
    st.divider()
    
    # --- NUEVO: MONITOR DE COSTOS ---
    st.header("💰 Monitor de Costos Totales")
    
    st.metric(label="Costo Acumulado (USD)", value=f"${st.session_state.costo_total['costo']:.4f}")
    
    col_a, col_b = st.columns(2)
    col_a.metric(label="Tokens Totales", value=st.session_state.costo_total["tokens"])
    col_b.metric(label="Consultas", value=st.session_state.costo_total["consultas"])
    
    with st.expander("Ver desglose de tokens"):
        st.write(f"- **📥 Entrada (Prompt):** {st.session_state.costo_total['tokens_entrada']}")
        st.write(f"- **📤 Salida (Generación):** {st.session_state.costo_total['tokens_salida']}")
        st.write(f"- **🔌 Llamadas API:** {st.session_state.costo_total['llamadas']}")
    
    st.divider()
    
    if st.button("🗑️ Limpiar Chat y Costos", type="secondary", use_container_width=True):
        st.session_state.messages = []
        # Reiniciar también los costos si limpia la conversación
        st.session_state.costo_total = {
            "tokens": 0, "tokens_entrada": 0, "tokens_salida": 0,
            "costo": 0.0, "llamadas": 0, "consultas": 0
        }
        st.rerun()

# Layout principal con columnas
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat")
    
    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # NUEVO: Mostrar el costo individual de cada respuesta
            if message["role"] == "assistant" and "costo" in message:
                with st.expander("💸 Costo de esta consulta", expanded=False):
                    c = message['costo']
                    st.caption(f"**Costo:** ${c['costo']:.5f} | **Tokens:** {c['tokens']} (Entrada: {c['tokens_entrada']} - Salida: {c['tokens_salida']})")

with col2:
    st.markdown("### 📄 Documentos Relevantes")
    
    # Mostrar documentos de la última consulta
    if st.session_state.messages:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "assistant" and "docs" in last_message:
            docs = last_message["docs"]
            
            if docs:
                for doc in docs:
                    with st.expander(f"📄 Fragmento {doc['fragmento']}", expanded=False):
                        st.markdown(f"**Fuente:** {doc['fuente']}")
                        st.markdown(f"**Página:** {doc['pagina']}")
                        st.markdown("**Contenido:**")
                        st.text(doc['contenido'])

# Input del usuario
if prompt := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generar respuesta
    with st.spinner("🔍 Analizando..."):
        response, docs, costo = query_rag(prompt)
    
        # Actualizar acumuladores globales
        st.session_state.costo_total["tokens"] += costo.get("tokens", 0)
        st.session_state.costo_total["tokens_entrada"] += costo.get("tokens_entrada", 0)
        st.session_state.costo_total["tokens_salida"] += costo.get("tokens_salida", 0)
        st.session_state.costo_total["llamadas"] += costo.get("llamadas", 0)
        st.session_state.costo_total["costo"] += costo.get("costo", 0.0)
        st.session_state.costo_total["consultas"] += 1
    
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "docs": docs,
            "costo": costo          
        })
    
    # Recargar para mostrar los nuevos mensajes
    st.rerun()

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666;'>🏛️ Asistente Legal</div>", 
    unsafe_allow_html=True
)