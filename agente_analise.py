# agente_analise.py

import os
import pandas as pd
from pandasql import sqldf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks import get_openai_callback 
from langchain_core.documents import Document 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil 
import io 
import streamlit as st 

print("DEBUG: bibliotecas importadas com sucesso")

load_dotenv()
chave = os.getenv("api_key") 

# *****************************************************************
# FUNÇÃO PRINCIPAL: Configura o Agente (Versão com upload)
# *****************************************************************

def configurar_agente(csv_data: bytes, txt_data: bytes):
    """
    Configura todo o pipeline do agente a partir de dados de arquivos em memória.
    """
    try:
        base_csv = pd.read_csv(io.BytesIO(csv_data))
        print(f"DEBUG: criado o dataframe base_csv com sucesso contendo {len(base_csv)} linhas")
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o arquivo CSV. Erro: {e}")
        raise

    txt_content = txt_data.decode("utf-8")
    documents = [Document(page_content=txt_content, metadata={"source": "upload_detalhamento"})]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(api_key=chave)
    
    vectorstore = None
    if splits:
        try:
            # MUDANÇA CRUCIAL: Inicia o ChromaDB em memória, sem persistir em disco
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            print("DEBUG: ChromaDB criado em memória com sucesso.")
        except Exception as e:
            print(f"ERRO: Falha crítica ao criar ChromaDB: {e}")
            vectorstore = None
    else:
        print("AVISO: RAG não configurado, pois 'splits' está vazio.")
        
    colunas = base_csv.columns 

    @tool
    def consulta_sql(query: str) -> str:
        """Sempre use esta ferramenta para consultar informações de tabelas estruturadas..."""
        try:
            result = sqldf(query, {'dados_tabela': base_csv})
            if result.empty:
                return "A consulta SQL não retornou resultados."
            return result.to_string(index=False)
        except Exception as e:
            return f"Erro na consulta SQL: {e}. Verifique a sintaxe da sua query ou o nome da tabela (use 'dados_tabela')."

    @tool
    def estatistica(tipo_calculo: str, coluna: str) -> str:
        """Use esta ferramenta para realizar **QUALQUER** cálculo estatístico via NumPy. Tipos de cálculo suportados: 'max', 'min', 'media', 'mediana', 'desvio_padrao' e 'variancia'. Você deve fornecer o 'tipo_calculo' e o nome da 'coluna' como argumentos."""
        if coluna not in base_csv.columns:
            return f"Erro: A coluna '{coluna}' não foi encontrada no DataFrame. Por favor, verifique a ortografia."
        
        dados = pd.to_numeric(base_csv[coluna], errors='coerce').dropna()
        
        if dados.empty:
            return f"Erro: A coluna '{coluna}' não contém dados numéricos válidos para o cálculo estatístico."

        try:
            if tipo_calculo.lower() == 'max':
                resultado = np.max(dados)
                return f"O valor máximo da coluna '{coluna}' é: {resultado}"
            elif tipo_calculo.lower() == 'min':
                resultado = np.min(dados)
                return f"O valor mínimo da coluna '{coluna}' é: {resultado}"
            elif tipo_calculo.lower() == 'media':
                resultado = np.mean(dados)
                return f"A média da coluna '{coluna}' é: {resultado}"
            elif tipo_calculo.lower() == 'mediana':
                resultado = np.median(dados)
                return f"A mediana da coluna '{coluna}' é: {resultado}"
            elif tipo_calculo.lower() == 'desvio_padrao':
                resultado = np.std(dados)
                return f"O desvio padrão da coluna '{coluna}' é: {resultado}"
            elif tipo_calculo.lower() == 'variancia':
                resultado = np.var(dados)
                return f"A variância da coluna '{coluna}' é: {resultado}"
            else:
                return "Cálculo não suportado. Escolha entre 'max', 'min', 'media', 'mediana', 'desvio_padrao' ou 'variancia'."
        except Exception as e:
            return f"Ocorreu um erro inesperado ao calcular {tipo_calculo} para a coluna '{coluna}': {e}"

    @tool
    def outlier(coluna: str) -> str:
        """Utilize essa ferramenta para encontrar outliers em uma coluna específica usando o método do Intervalo Interquartil (IQR)."""
        if coluna not in base_csv.columns:
            return f"Erro: A coluna '{coluna}' não foi encontrada no DataFrame. Por favor, verifique a ortografia."
        try:
            dados = pd.to_numeric(base_csv[coluna], errors='coerce').dropna()
            
            if dados.empty:
                return f"Erro: A coluna '{coluna}' não contém dados numéricos válidos para encontrar outliers."
                
            Q1 = np.quantile(dados, 0.25)
            Q3 = np.quantile(dados, 0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            outliers_encontrados = dados[(dados < lower_limit) | (dados > upper_limit)]
            if outliers_encontrados.empty:
                return f"Nenhum outlier encontrado na coluna '{coluna}'."
            else:
                total_outliers = len(outliers_encontrados)
                return f"Outliers encontrados na coluna '{coluna}': {total_outliers} valores."
        except Exception as e:
            return f"Ocorreu um erro inesperado ao processar a coluna '{coluna}': {e}"

    @tool
    def correlacao(metodo="pearson"):
        """Utilize esta ferramenta para encontrar uma correlação existente entre as colunas de um dataframe, entendendo a influência entre elas."""
        matriz_corr = base_csv.corr(method=metodo, numeric_only=True)
        return matriz_corr.to_string() 

    @tool
    def criar_grafico(tipo_grafico: str, coluna_x: str, coluna_y: str = None, titulo: str = "Gráfico Gerado") -> str:
        """Gera um gráfico a partir dos dados do dataframe. Tipos de gráfico suportados: 'barras', 'pizza', 'histograma' e 'dispersao'."""
        try:
            if coluna_x not in base_csv.columns:
                return f"Erro: Coluna '{coluna_x}' não encontrada no DataFrame."
            if coluna_y and coluna_y not in base_csv.columns:
                return f"Erro: Coluna '{coluna_y}' não encontrada no DataFrame."
            
            fig = plt.figure(figsize=(10, 6))
            
            if tipo_grafico.lower() == 'barras':
                if not coluna_y: return "Erro: Gráfico de barras requer 'coluna_y' para os valores."
                plt.bar(base_csv[coluna_x], base_csv[coluna_y])
                plt.xlabel(coluna_x.replace('_', ' ').title())
                plt.ylabel(coluna_y.replace('_', ' ').title())
                plt.xticks(rotation=45, ha='right')
            elif tipo_grafico.lower() == 'pizza':
                labels = base_csv[coluna_y] if coluna_y and coluna_y in base_csv.columns else base_csv[coluna_x].index
                plt.pie(base_csv[coluna_x].value_counts(), labels=labels, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
            elif tipo_grafico.lower() == 'histograma':
                plt.hist(pd.to_numeric(base_csv[coluna_x], errors='coerce').dropna(), bins=10, edgecolor='black')
                plt.xlabel(coluna_x.replace('_', ' ').title())
                plt.ylabel('Frequência')
            elif tipo_grafico.lower() == 'dispersao':
                if not coluna_y: return "Erro: Gráfico de dispersão requer 'coluna_y'."
                plt.scatter(base_csv[coluna_x], base_csv[coluna_y])
                plt.xlabel(coluna_x.replace('_', ' ').title())
                plt.ylabel(coluna_y.replace('_', ' ').title())
            else:
                return "Tipo de gráfico não suportado."

            plt.title(titulo)
            plt.tight_layout()
            st.session_state.grafico_para_exibir = fig
            return f"Gráfico do tipo {tipo_grafico} para as colunas {coluna_x} e {coluna_y or 'contagem'} gerado com sucesso! O gráfico será exibido abaixo."

        except Exception as e:
            st.session_state.grafico_para_exibir = None
            return f"Erro ao criar gráfico: {e}. Verifique se as colunas existem e os dados são do tipo correto."

    @tool
    def buscar_documentos(query: str) -> str:
        """Busca informações relevantes na base de conhecimento do dataframe no arquivo .txt)..."""
        if vectorstore is None:
            return "A base de conhecimento não foi configurada. Não é possível buscar documentos."
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(query)
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        else:
            return "Não encontrei informações relevantes nos documentos para esta consulta."
            
    main_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5 
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.25,
        api_key=chave,
        max_tokens=2000
    )

    tools = [criar_grafico, consulta_sql, correlacao, buscar_documentos, estatistica, outlier]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """você é um especialista em estatística e dados..."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=main_memory,
        verbose=False,
        handle_parsing_errors=True
    )

    print("DEBUG: agente configurado com sucesso")
    return agent_executor