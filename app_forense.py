import streamlit as st
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO
import time
from datetime import datetime, timedelta
import os
from fpdf import FPDF
import numpy as np

# Configuração da Página
st.set_page_config(page_title="Polícia Civil - Sistema Forense", page_icon="🕵️", layout="wide")

# --- DICIONÁRIO DE TRADUÇÃO ---
tradutor = {
    'person': 'Pessoa', 'bicycle': 'Bicicleta', 'car': 'Carro', 'motorcycle': 'Moto',
    'bus': 'Ônibus', 'truck': 'Caminhão', 'knife': 'Faca', 'pistol': 'Pistola',
    'rifle': 'Fuzil', 'handgun': 'Arma de Mão', 'backpack': 'Mochila',
    'handbag': 'Bolsa', 'suitcase': 'Mala', 'cell phone': 'Celular'
}

# --- FUNÇÃO DE DETECÇÃO DE COR (NOVA) ---
def detectar_cor_dominante(frame, box):
    """
    Recorta o objeto da imagem e define a cor predominante baseada em distância Euclidiana.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Proteção para não cortar fora da imagem
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    recorte = frame[y1:y2, x1:x2]
    
    if recorte.size == 0: return ""

    # Calcula a média de cor do recorte (BGR)
    # Pegamos apenas o centro da imagem para evitar pegar o fundo (asfalto, calçada)
    h_rec, w_rec, _ = recorte.shape
    centro_x, centro_y = w_rec // 2, h_rec // 2
    margem_x, margem_y = w_rec // 4, h_rec // 4 # Pega 50% central
    
    recorte_central = recorte[centro_y-margem_y:centro_y+margem_y, centro_x-margem_x:centro_x+margem_x]
    
    if recorte_central.size == 0:
        media_bgr = np.mean(recorte, axis=(0, 1))
    else:
        media_bgr = np.mean(recorte_central, axis=(0, 1))

    # Cores de Referência (BGR)
    cores_referencia = {
        'Preto': (0, 0, 0),
        'Branco': (255, 255, 255),
        'Cinza': (128, 128, 128),
        'Vermelho': (0, 0, 255),
        'Verde': (0, 255, 0),
        'Azul': (255, 0, 0),
        'Amarelo': (0, 255, 255),
        'Laranja': (0, 165, 255) # OpenCV usa BGR
    }

    menor_distancia = float('inf')
    nome_cor = "Indefinido"

    for nome, valor in cores_referencia.items():
        # Distância Euclidiana simples
        dist = np.linalg.norm(media_bgr - np.array(valor))
        if dist < menor_distancia:
            menor_distancia = dist
            nome_cor = nome

    return nome_cor

# --- CLASSE DO RELATÓRIO PDF ---
class RelatorioPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'POLICIA CIVIL - RELATORIO DE ANALISE DE MIDIA', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def gerar_pdf_inquerito(dados, estatisticas, nome_video, usuario):
    pdf = RelatorioPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    # Cabeçalho
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Referencia: Analise Automatizada de Video", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, f"Data da Analise: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1)
    pdf.cell(0, 5, f"Agente Responsavel: {usuario}", 0, 1)
    pdf.cell(0, 5, f"Arquivo Analisado: {nome_video}", 0, 1)
    pdf.ln(10)

    # Resumo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. RESUMO DOS ELEMENTOS IDENTIFICADOS", 0, 1)
    pdf.set_font("Arial", size=10)
    
    texto_resumo = ""
    for obj, qtd in estatisticas.items():
        texto_resumo += f"- {obj}: {qtd} deteccoes\n"
    pdf.multi_cell(0, 7, texto_resumo)
    pdf.ln(5)

    # Tabela Detalhada
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. LINHA DO TEMPO E DETALHES", 0, 1)
    
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", 'B', 9)
    # Ajuste de largura das colunas
    pdf.cell(25, 8, "Tempo", 1, 0, 'C', 1)
    pdf.cell(50, 8, "Objeto", 1, 0, 'L', 1)
    pdf.cell(50, 8, "Detalhes Visuais (Cor)", 1, 0, 'L', 1)
    pdf.cell(25, 8, "Confianca", 1, 1, 'C', 1)
    
    pdf.set_font("Arial", size=9)
    for item in dados:
        alvo = item['Alvo'].encode('latin-1', 'ignore').decode('latin-1')
        detalhe = item['Detalhes'].encode('latin-1', 'ignore').decode('latin-1')
        
        pdf.cell(25, 7, item['Minuto'], 1, 0, 'C')
        pdf.cell(50, 7, alvo, 1, 0, 'L')
        pdf.cell(50, 7, detalhe, 1, 0, 'L')
        pdf.cell(25, 7, item['Conf'], 1, 1, 'C')

    pdf.ln(20)
    pdf.cell(0, 5, "_"*60, 0, 1, 'C')
    pdf.cell(0, 5, f"Agente: {usuario}", 0, 1, 'C')

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

    st.sidebar.header("⚙️ Parâmetros")
    model_choice = st.sidebar.radio("Modelo de IA:", ["Padrão", "Customizado"])
    model_path = 'yolov8n.pt'
    if model_choice == "Customizado":
        model_file = st.sidebar.file_uploader("Carregar Modelo (.pt)", type=['pt'])
        if model_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
                tmp_model.write(model_file.read())
                model_path = tmp_model.name
    
    conf_threshold = st.sidebar.slider("Confiança Mínima (%)", 0, 100, 45) / 100

    uploaded_video = st.file_uploader("📂 Carregar Vídeo do Inquérito", type=['mp4', 'avi', 'mov'])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Visualização")
            st_frame = st.empty()
        with col2:
            st.subheader("Ocorrências Detalhadas")
            log_placeholder = st.empty()
            
        if st.button("▶️ INICIAR VARREDURA DETALHADA", type="primary"):
            try:
                model = YOLO(model_path)
            except:
                st.warning("Usando modelo padrão...")
                model = YOLO('yolov8n.pt')

            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30
            
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
                
                # Cópia para desenhar
                frame_anotado = frame.copy()
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        
                        # Desenha a caixa padrão da IA
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        if cls_id < len(model.names):
                            nome_ingles = model.names[cls_id]
                            nome_pt = tradutor.get(nome_ingles, nome_ingles)
                            
                            # --- DETECÇÃO DE DETALHES (COR) ---
                            cor_predominante = detectar_cor_dominante(frame, box)
                            descricao_completa = f"{nome_pt} ({cor_predominante})"
                            
                            # Filtro de repetição (2s)
                            if tempo_seg - ultimo_registro.get(descricao_completa, -10) > 2.0:
                                log_data.append({
                                    "Minuto": tempo_fmt, 
                                    "Alvo": nome_pt, 
                                    "Detalhes": cor_predominante,
                                    "Conf": f"{float(box.conf[0]):.2f}"
                                })
                                ultimo_registro[descricao_completa] = tempo_seg
                                estatisticas[nome_pt] = estatisticas.get(nome_pt, 0) + 1
                                
                            # Escreve na tela também
                            label = f"{nome_pt} [{cor_predominante}]"
                            cv2.putText(frame_anotado, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
                
                if log_data:
                    # Mostra tabela simplificada na tela
                    df_view = pd.DataFrame(log_data)
                    log_placeholder.dataframe(df_view.iloc[::-1].head(8), hide_index=True)

                if total_frames > 0:
                    current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    progress_bar.progress(min(current / total_frames, 1.0))

            cap.release()
            
            if log_data:
                st.success("Varredura Finalizada.")
                
                # --- ÁREA DE DOWNLOADS ---
                st.markdown("### 📄 Documentação Oficial")
                col_d1, col_d2 = st.columns(2)
                
                df = pd.DataFrame(log_data)
                col_d1.download_button("Baixar CSV Bruto", df.to_csv(index=False), "dados.csv")
                
                pdf_bytes = gerar_pdf_inquerito(
                    log_data, 
                    estatisticas, 
                    uploaded_video.name, 
                    st.session_state.get('usuario_nome', 'POLICIAL')
                )
                
                col_d2.download_button(
                    label="📥 BAIXAR LAUDO DETALHADO (PDF)",
                    data=pdf_bytes,
                    file_name="Laudo_Detalhado.pdf",
                    mime="application/pdf"
                )

if check_password():
    app_principal()
else:
    tela_login()
