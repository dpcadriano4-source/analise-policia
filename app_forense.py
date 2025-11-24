import streamlit as st
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO
import time
from datetime import timedelta
import os

# Configuração da Página
st.set_page_config(page_title="Polícia Civil - Sistema Forense", page_icon="🕵️", layout="wide")

# --- SISTEMA DE LOGIN ---
def check_password():
    """Retorna True se o usuário estiver logado, False caso contrário."""
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
            # CREDENCIAIS (Para teste. Em produção use banco de dados)
            if usuario == "policial" and senha == "policia123":
                st.session_state['logado'] = True
                st.rerun()
            else:
                st.error("Acesso Negado. Credenciais inválidas.")

# --- O APLICATIVO PRINCIPAL ---
def app_principal():
    # Botão de Logout na Sidebar
    if st.sidebar.button("Sair / Logout"):
        st.session_state['logado'] = False
        st.rerun()

    st.title("🕵️ Sistema de Análise de Vídeo Forense")
    st.info("Usuário Logado: Agente Policial")
    st.markdown("---")

    # --- BARRA LATERAL (CONFIGURAÇÕES) ---
    st.sidebar.header("⚙️ Parâmetros")

    # Upload de Modelo
    model_choice = st.sidebar.radio("Modelo de IA:", ["Padrão (Pessoas/Veículos)", "Customizado (Armas/Drogas)"])
    model_path = 'yolov8n.pt' # Padrão
    
    if model_choice == "Customizado (Armas/Drogas)":
        model_file = st.sidebar.file_uploader("Carregar Modelo (.pt)", type=['pt'])
        if model_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
                tmp_model.write(model_file.read())
                model_path = tmp_model.name
    
    conf_threshold = st.sidebar.slider("Confiança Mínima (%)", 0, 100, 45) / 100

    # --- ÁREA DE UPLOAD ---
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
                st.warning("Baixando modelo padrão...")
                model = YOLO('yolov8n.pt')

            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            log_data = []
            ultimo_registro = {}
            
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # IA
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                
                # Logica de Log
                tempo_seg = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                tempo_fmt = str(timedelta(seconds=int(tempo_seg)))
                
                # Desenhar
                frame_anotado = results[0].plot()
                
                # Preencher dados
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        # Proteção contra erro de índice se mudar modelo
                        if cls_id < len(model.names):
                            nome_obj = model.names[cls_id]
                            
                            # Filtro de repetição (3s)
                            if tempo_seg - ultimo_registro.get(nome_obj, -10) > 3.0:
                                log_data.append({"Minuto": tempo_fmt, "Alvo": nome_obj, "Conf": f"{float(box.conf[0]):.2f}"})
                                ultimo_registro[nome_obj] = tempo_seg

                # Renderizar
                frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Tabela
                if log_data:
                    log_placeholder.dataframe(pd.DataFrame(log_data).iloc[::-1].head(8), hide_index=True)

                # Progresso
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if total_frames > 0:
                    progress_bar.progress(min(current_frame / total_frames, 1.0))

            cap.release()
            
            if log_data:
                st.success("Varredura Finalizada.")
                df = pd.DataFrame(log_data)
                st.download_button("Baixar Relatório (Excel/CSV)", df.to_csv(index=False), "laudo.csv")

# --- CONTROLE DE FLUXO ---
if check_password():
    app_principal()
else:
    tela_login()
