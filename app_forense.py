import streamlit as st
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO
import time
from datetime import datetime, timedelta
import os
from fpdf import FPDF

# Configuração da Página
st.set_page_config(page_title="Polícia Civil - Sistema Forense", page_icon="🕵️", layout="wide")

# --- DICIONÁRIO DE TRADUÇÃO ---
tradutor = {
    'person': 'Pessoa', 'bicycle': 'Bicicleta', 'car': 'Carro', 'motorcycle': 'Moto',
    'bus': 'Ônibus', 'truck': 'Caminhão', 'knife': 'Faca', 'pistol': 'Pistola',
    'rifle': 'Fuzil', 'handgun': 'Arma de Mão', 'backpack': 'Mochila'
}

# --- CLASSE DO RELATÓRIO PDF ---
class RelatorioPDF(FPDF):
    def header(self):
        # Título
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'POLICIA CIVIL - RELATORIO DE ANALISE DE MIDIA', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        # Rodapé com numeração
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def gerar_pdf_inquerito(dados, estatisticas, nome_video, usuario):
    pdf = RelatorioPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    # 1. Cabeçalho do Inquérito
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Referencia: Analise Automatizada de Video", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, f"Data da Analise: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1)
    pdf.cell(0, 5, f"Agente Responsavel: {usuario}", 0, 1)
    pdf.cell(0, 5, f"Arquivo Analisado: {nome_video}", 0, 1)
    pdf.ln(10)

    # 2. Resumo Estatístico
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. RESUMO DOS ELEMENTOS IDENTIFICADOS", 0, 1)
    pdf.set_font("Arial", size=10)
    
    texto_resumo = ""
    for obj, qtd in estatisticas.items():
        texto_resumo += f"- {obj}: {qtd} ocorrencia(s)\n"
    
    pdf.multi_cell(0, 7, texto_resumo)
    pdf.ln(5)

    # 3. Tabela Detalhada
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. LINHA DO TEMPO DETALHADA", 0, 1)
    
    # Cabeçalho da tabela
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(30, 8, "Tempo", 1, 0, 'C', 1)
    pdf.cell(100, 8, "Elemento Identificado", 1, 0, 'L', 1)
    pdf.cell(30, 8, "Confianca", 1, 1, 'C', 1)
    
    # Linhas da tabela
    pdf.set_font("Arial", size=10)
    for item in dados:
        # Tratamento simples para caracteres especiais no PDF (latin-1)
        alvo = item['Alvo'].encode('latin-1', 'ignore').decode('latin-1')
        pdf.cell(30, 7, item['Minuto'], 1, 0, 'C')
        pdf.cell(100, 7, alvo, 1, 0, 'L')
        pdf.cell(30, 7, item['Conf'], 1, 1, 'C')

    pdf.ln(20)
    
    # 4. Assinatura
    pdf.cell(0, 5, "_"*60, 0, 1, 'C')
    pdf.cell(0, 5, f"Agente: {usuario}", 0, 1, 'C')
    pdf.cell(0, 5, "Assinatura Digital / Matricula", 0, 1, 'C')

    return pdf.output(dest='S').encode('latin-1')

# --- SISTEMA DE LOGIN ---
def check_password():
    if 'logado' not in st.session_state:
        st.session_state['logado'] = False
    return st.session_state['logado']

def tela_login():
    st.markdown("<h1 style='text-align: center;'>🔐 Acesso Restrito - Investigação</h1>", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        usuario = st.text_input("Usuário")
        senha = st.text_input("Senha", type="password")
        if st.button("Entrar no Sistema"):
            if usuario == "policial" and senha == "policia123":
                st.session_state['logado'] = True
                st.session_state['usuario_nome'] = usuario.upper()
                st.rerun()
            else:
                st.error("Acesso Negado.")

# --- O APLICATIVO PRINCIPAL ---
def app_principal():
    if st.sidebar.button("Sair / Logout"):
        st.session_state['logado'] = False
        st.rerun()

    st.title("🕵️ Sistema de Análise de Vídeo Forense")
    st.info(f"Agente Logado: {st.session_state.get('usuario_nome', 'POLICIAL')}")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Parâmetros")
    model_choice = st.sidebar.radio("Modelo de IA:", ["Padrão (Pessoas/Veículos)", "Customizado (Armas/Drogas)"])
    model_path = 'yolov8n.pt'
    if model_choice == "Customizado (Armas/Drogas)":
        model_file = st.sidebar.file_uploader("Carregar Modelo (.pt)", type=['pt'])
        if model_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
                tmp_model.write(model_file.read())
                model_path = tmp_model.name
    conf_threshold = st.sidebar.slider("Confiança Mínima (%)", 0, 100, 45) / 100

    # Upload
    uploaded_video = st.file_uploader("📂 Carregar Vídeo do Inquérito", type=['mp4', 'avi', 'mov'])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Visualização")
            st_frame = st.empty()
        with col2:
            st.subheader("Ocorrências Detectadas")
            log_placeholder = st.empty()
            
        if st.button("▶️ INICIAR VARREDURA", type="primary"):
            try:
                model = YOLO(model_path)
            except:
                st.warning("Usando modelo padrão...")
                model = YOLO('yolov8n.pt')

            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            log_data = []
            estatisticas = {}
            ultimo_registro = {}
            
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                tempo_seg = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                tempo_fmt = str(timedelta(seconds=int(tempo_seg)))
                frame_anotado = results[0].plot()
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id < len(model.names):
                            nome_ingles = model.names[cls_id]
                            nome_pt = tradutor.get(nome_ingles, nome_ingles)
                            
                            # Filtro de repetição (2s)
                            if tempo_seg - ultimo_registro.get(nome_pt, -10) > 2.0:
                                log_data.append({"Minuto": tempo_fmt, "Alvo": nome_pt, "Conf": f"{float(box.conf[0]):.2f}"})
                                ultimo_registro[nome_pt] = tempo_seg
                                # Atualiza estatísticas totais
                                estatisticas[nome_pt] = estatisticas.get(nome_pt, 0) + 1

                frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
                
                if log_data:
                    log_placeholder.dataframe(pd.DataFrame(log_data).iloc[::-1].head(8), hide_index=True)

                if total_frames > 0:
                    current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    progress_bar.progress(min(current / total_frames, 1.0))

            cap.release()
            
            if log_data:
                st.success("Varredura Finalizada com Sucesso.")
                
                # --- ÁREA DE DOWNLOADS ---
                st.markdown("### 📄 Documentação Oficial")
                col_d1, col_d2 = st.columns(2)
                
                # Botão 1: CSV (Dados Brutos)
                df = pd.DataFrame(log_data)
                col_d1.download_button("Baixar Dados (Excel/CSV)", df.to_csv(index=False), "dados_brutos.csv")
                
                # Botão 2: PDF (Laudo Formatado)
                pdf_bytes = gerar_pdf_inquerito(
                    log_data, 
                    estatisticas, 
                    uploaded_video.name, 
                    st.session_state.get('usuario_nome', 'POLICIAL')
                )
                
                col_d2.download_button(
                    label="📥 BAIXAR LAUDO TÉCNICO (PDF)",
                    data=pdf_bytes,
                    file_name="Laudo_Investigacao.pdf",
                    mime="application/pdf"
                )

if check_password():
    app_principal()
else:
    tela_login()
