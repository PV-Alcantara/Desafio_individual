# app_streamlit.py

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

# 1. Importa a função de configuração do agente
from agente_analise import configurar_agente 

# --- Configuração da Página ---
st.set_page_config(page_title="Agente de Análise de Dados Financeiros", layout="wide")
st.title("📊 Agente de Análise de Dados CSV")
st.caption("Um especialista em estatística e dados, pronto para analisar seu arquivo CSV.")
st.divider()

# --- UPLOAD DE ARQUIVOS ---
st.header("Upload de Arquivos")
st.warning("⚠️ O agente só será configurado após o upload dos dois arquivos.")

csv_file = st.file_uploader(
    "1. Escolha o arquivo de dados (CSV)", 
    type=['csv'], 
    key="csv_upload"
)

txt_file = st.file_uploader(
    "2. Escolha o arquivo de detalhamento (TXT)", 
    type=['txt'], 
    key="txt_upload"
)

# --- Cache e Inicialização do Agente (DEPENDENTE DOS UPLOADS) ---

@st.cache_resource
def get_agent_executor(csv_data, txt_data):
    """
    Configura e retorna o agente, usando os dados dos arquivos.
    O cache garante que a função só será re-executada se os dados mudarem.
    """
    try:
        return configurar_agente(csv_data, txt_data)
    except Exception as e:
        st.error(f"Falha ao inicializar o agente. Verifique a API Key ou o formato dos arquivos. Erro: {e}")
        st.stop()

# A execução do script é interrompida aqui se os arquivos não foram carregados
agent_executor = None
if csv_file is not None and txt_file is not None:
    csv_bytes = csv_file.getvalue()
    txt_bytes = txt_file.getvalue()
    agent_executor = get_agent_executor(csv_bytes, txt_bytes)
    st.success("✅ Arquivos carregados e Agente configurado! Você já pode conversar.")
    st.divider()
else:
    st.info("Aguardando o upload dos arquivos...")
    st.stop() 

# --- Gerenciamento do Histórico de Conversa e Estado do Gráfico ---

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Olá! Sou seu Agente de Análise. Faça perguntas pertinentes a base de dados carregada."
    })

if 'grafico_para_exibir' not in st.session_state:
    st.session_state.grafico_para_exibir = None

# 2. Exibir o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and 'figura_anexada' in message:
            fig = message['figura_anexada']
            st.pyplot(fig) 
            plt.close(fig)

# --- Captura de Input e Execução do Agente ---

if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🚀 Analisando dados, buscando documentos e pensando na resposta..."):
            with get_openai_callback() as cb:
                try:
                    response = agent_executor.invoke({"input": prompt})
                    ai_response_text = response["output"]
                except Exception as e:
                    ai_response_text = f"Ocorreu um erro durante a execução: {e}"
                    st.error(ai_response_text)
            
            fig = st.session_state.grafico_para_exibir 
            
            ai_message = {"role": "assistant", "content": ai_response_text}
            
            if fig is not None:
                ai_message['figura_anexada'] = fig
                st.session_state.grafico_para_exibir = None 

            st.markdown(ai_response_text)

            with st.expander("Detalhes da Execução"):
                st.write(f"Total de tokens consumidos: **{cb.total_tokens}**")
                st.write(f"Custo total em USD: **${cb.total_cost:.6f}**")
    
    st.session_state.messages.append(ai_message)
    st.rerun()