# credit_app.py
import os
import re
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
from anchor import anchor_tabular
import lime.lime_tabular
import eli5
from eli5 import format_as_html
from openai import OpenAI
from dotenv import load_dotenv

# --- NOVA FUNCIONALIDADE: Dependências para Geração de PDF --- #
import base64
import io
from xhtml2pdf import pisa
from PIL import Image
import markdown

# ------------------ Utilitários (sem alteração) ------------------ #
def format_currency(value):
    """Formata número em R$ 1.234.567,89. Seguro contra None/NaN."""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return "R$ 0,00"
    s = f"R$ {v:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def format_shap_contrib(value):
    """Formata contribuição SHAP como número com ponto decimal (ex: -3.24)."""
    try:
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return f"{value}"

def _format_number_in_rule(num_str, is_money):
    try:
        v = float(num_str)
        if is_money:
            return format_currency(v)
        return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return num_str

def humanize_lime_rule(rule, feature_translations, input_values):
    clauses = re.split(r'\s+and\s+|\s*&\s*', rule, flags=re.IGNORECASE)
    human_clauses = []
    feature_name = None

    for c in clauses:
        c = c.strip().strip('()')
        match = re.search(r'([A-Za-z_][A-Za-z0-9_]*)', c)
        if match:
            feature_name = match.group(1)
            is_money = feature_name in ['VL_IMOVEIS', 'VALOR_TABELA_CARROS', 'ULTIMO_SALARIO', 'OUTRA_RENDA_VALOR']
            translated_feature = feature_translations.get(feature_name, feature_name)

            nums = re.findall(r'([-+]?\d*\.?\d+)', c)
            formatted_c = c
            for num in nums:
                formatted_c = formatted_c.replace(num, _format_number_in_rule(num, is_money), 1)
            
            # Substitui o nome técnico pelo humanizado
            formatted_c = formatted_c.replace(feature_name, translated_feature)
            human_clauses.append(formatted_c)
        else:
            human_clauses.append(c)
    
    humanized = " and ".join(human_clauses)
    input_value_str = None
    if feature_name and feature_name in input_values:
        val = input_values[feature_name]
        is_money_input = feature_name in ['VL_IMOVEIS', 'VALOR_TABELA_CARROS', 'ULTIMO_SALARIO', 'OUTRA_RENDA_VALOR']
        input_value_str = format_currency(val) if is_money_input else str(val)

    return humanized, input_value_str

# --- NOVA FUNCIONALIDADE: Geração de Relatório PDF --- #

# --- FUNÇÃO ATUALIZADA ---
def create_pdf_report(result_text, proba, shap_fig, shap_reasons, lime_reasons, llm_feedback):
    """Gera um relatório PDF a partir dos resultados da análise."""
    
    # 1. Converter a figura SHAP para uma imagem em base64
    buf = io.BytesIO()
    shap_fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
    shap_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # 2. !! IMPORTANTE: Converter o feedback do LLM de Markdown para HTML !!
    llm_feedback_html = markdown.markdown(llm_feedback)

    # 3. Montar o conteúdo HTML do relatório
    html = f"""
    <html>
    <head>
        <style>
            @page {{
                size: a4 portrait;
                margin: 1.5cm;
            }}
            body {{
                font-family: 'Helvetica', 'Arial', sans-serif;
                color: #333;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                font-family: 'Georgia', serif;
            }}
            h1 {{
                color: #003366;
                text-align: center;
                border-bottom: 2px solid #003366;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #0055A4;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
                margin-top: 25px;
            }}
            h3 {{
                color: #333;
                font-size: 1.1em;
            }}
            .result {{
                font-size: 1.2em;
                font-weight: bold;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: {'#28a745' if result_text == 'Approved' else '#dc3545'};
            }}
            .probability {{
                text-align: center;
                font-size: 1.1em;
                margin-bottom: 20px;
            }}
            .explanation-section ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            .explanation-section li {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 10px;
                margin-bottom: 8px;
            }}
            .shap-image {{
                text-align: center;
                margin-top: 20px;
            }}
            .llm-feedback ul {{
                list-style-position: inside;
                padding-left: 20px;
            }}
            .llm-feedback p, .llm-feedback li {{
                margin-bottom: 10px;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <h1>Credit Analysis Report</h1>
        
        <h2>Prediction Result</h2>
        <div class="result">{result_text}</div>
        <div class="probability">Approval Probability: <strong>{proba:.2%}</strong></div>

        <h2>SHAP Explanation (Feature Impact)</h2>
        <div class="explanation-section">
            <ul>{''.join([f'<li>{reason}</li>' for reason in shap_reasons])}</ul>
        </div>
        <div class="shap-image">
            <img src="data:image/png;base64,{shap_img_base64}" />
        </div>

        <h2>LIME Explanation (Local Rules)</h2>
        <div class="explanation-section">
             <ul>{''.join([f'<li>{reason}</li>' for reason in lime_reasons])}</ul>
        </div>

        <h2>Expert Feedback (AI Generated)</h2>
        <div class="llm-feedback">
            {llm_feedback_html}
        </div>
    </body>
    </html>
    """
    
    # 4. Converter HTML para PDF
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html), dest=pdf_buffer)
    
    if pisa_status.err:
        return None
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# def create_pdf_report(result_text, proba, shap_fig, shap_reasons, lime_reasons, llm_feedback):
#     """Gera um relatório PDF a partir dos resultados da análise."""
    
#     # 1. Converter a figura SHAP para uma imagem em base64
#     buf = io.BytesIO()
#     shap_fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
#     shap_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#     buf.close()

#     # 2. Montar o conteúdo HTML do relatório
#     html = f"""
#     <html>
#     <head>
#         <style>
#             @page {{
#                 size: a4 portrait;
#                 margin: 1.5cm;
#             }}
#             body {{
#                 font-family: 'Helvetica', 'Arial', sans-serif;
#                 color: #333;
#             }}
#             h1 {{
#                 color: #003366;
#                 text-align: center;
#                 border-bottom: 2px solid #003366;
#                 padding-bottom: 10px;
#             }}
#             h2 {{
#                 color: #0055A4;
#                 border-bottom: 1px solid #ccc;
#                 padding-bottom: 5px;
#                 margin-top: 25px;
#             }}
#             .result {{
#                 font-size: 1.2em;
#                 font-weight: bold;
#                 padding: 10px;
#                 margin: 10px 0;
#                 border-radius: 5px;
#                 text-align: center;
#                 color: white;
#                 background-color: {'#28a745' if result_text == 'Approved' else '#dc3545'};
#             }}
#             .probability {{
#                 text-align: center;
#                 font-size: 1.1em;
#                 margin-bottom: 20px;
#             }}
#             .explanation-section ul {{
#                 list-style-type: none;
#                 padding-left: 0;
#             }}
#             .explanation-section li {{
#                 background-color: #f8f9fa;
#                 border: 1px solid #dee2e6;
#                 border-radius: 4px;
#                 padding: 10px;
#                 margin-bottom: 8px;
#             }}
#             .shap-image {{
#                 text-align: center;
#                 margin-top: 20px;
#             }}
#             img {{
#                 max-width: 100%;
#                 height: auto;
#             }}
#         </style>
#     </head>
#     <body>
#         <h1>Credit Analysis Report</h1>
        
#         <h2>Prediction Result</h2>
#         <div class="result">{result_text}</div>
#         <div class="probability">Approval Probability: <strong>{proba:.2%}</strong></div>

#         <h2>SHAP Explanation (Feature Impact)</h2>
#         <div class="explanation-section">
#             <ul>{''.join([f'<li>{reason}</li>' for reason in shap_reasons])}</ul>
#         </div>
#         <div class="shap-image">
#             <img src="data:image/png;base64,{shap_img_base64}" />
#         </div>

#         <h2>LIME Explanation (Local Rules)</h2>
#         <div class="explanation-section">
#              <ul>{''.join([f'<li>{reason}</li>' for reason in lime_reasons])}</ul>
#         </div>

#         <h2>Expert Feedback (AI Generated)</h2>
#         <div>{llm_feedback.replace('\\n', '<br>')}</div>

#     </body>
#     </html>
#     """
    
#     # 3. Converter HTML para PDF
#     pdf_buffer = io.BytesIO()
#     pisa_status = pisa.CreatePDF(io.StringIO(html), dest=pdf_buffer)
    
#     if pisa_status.err:
#         return None
    
#     pdf_buffer.seek(0)
#     return pdf_buffer.getvalue()


# ------------------ UI Config e Mapeamentos (Rótulos Traduzidos) ------------------ #
st.set_page_config(page_title="XAI Credit Analysis", layout="wide")
st.title("Creditworthiness Prediction and Explainability (XAI)")
st.markdown("Enter the applicant's details below to get a credit prediction and an explanation of the result.")


# Mapeamentos (valores em PT para o modelo)
ufs = ['SP', 'MG', 'SC', 'PR', 'RJ']
escolaridades = ['Superior Cursando', 'Superior Completo', 'Segundo Grau Completo']
estados_civis = ['Solteiro', 'Casado', 'Divorciado']
faixas_etarias = ['18-25', '26-35', '36-45', '46-60', 'Acima de 60']

uf_map = {label: i for i, label in enumerate(ufs)}
escolaridade_map = {label: i for i, label in enumerate(escolaridades)}
estado_civil_map = {label: i for i, label in enumerate(estados_civis)}
faixa_etaria_map = {label: i for i, label in enumerate(faixas_etarias)}

feature_names = [
    'UF', 'ESCOLARIDADE', 'ESTADO_CIVIL', 'QT_FILHOS', 'CASA_PROPRIA', 'QT_IMOVEIS', 
    'VL_IMOVEIS', 'OUTRA_RENDA', 'OUTRA_RENDA_VALOR', 'TEMPO_ULTIMO_EMPREGO_MESES', 
    'TRABALHANDO_ATUALMENTE', 'ULTIMO_SALARIO', 'QT_CARROS', 'VALOR_TABELA_CARROS', 'FAIXA_ETARIA'
]

# ------------------ Carregar Modelos e Dados ------------------ #
try:
    scaler = joblib.load('scaler.pkl')
    lr_model = joblib.load('modelo_regressao.pkl')
    X_train_raw = joblib.load('X_train.pkl')
    if isinstance(X_train_raw, np.ndarray):
        X_train_df = pd.DataFrame(X_train_raw, columns=feature_names)
    else:
        X_train_df = X_train_raw[feature_names] if list(X_train_raw.columns) != feature_names else X_train_raw
except Exception as e:
    st.error(f"Error loading models/data: {e}")
    st.stop()

# ------------------ OPENAI Client ------------------ #
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.warning("⚠️ OpenAI API key not configured. LLM feedback will be unavailable.")

# ------------------ Inputs da UI (Layout Refatorado) ------------------ #
with st.expander("Enter Applicant's Profile Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Personal & Financial Info")
        UF = st.selectbox('State (UF)', ufs, index=0)
        ESCOLARIDADE = st.selectbox('Education Level', escolaridades, index=1)
        ESTADO_CIVIL = st.selectbox('Marital Status', estados_civis, index=0)
        QT_FILHOS = st.number_input('Number of Children', min_value=0, value=1)
        FAIXA_ETARIA = st.radio('Age Group', faixas_etarias, index=2, horizontal=True)

    with col2:
        st.subheader("Assets & Employment")
        CASA_PROPRIA = st.radio('Owns a Home?', ['Sim', 'Não'], index=0, horizontal=True)
        if CASA_PROPRIA == 'Sim':
            QT_IMOVEIS = st.number_input('Number of Properties', min_value=1, value=1)
            VL_IMOVEIS = st.number_input('Total Value of Properties (R$)', min_value=0.0, value=100000.0, step=10000.0)
        else:
            QT_IMOVEIS = 0
            VL_IMOVEIS = 0.0
        
        TRABALHANDO_ATUALMENTE = st.radio('Currently Employed?', ['Sim', 'Não'], index=0, horizontal=True)
        if TRABALHANDO_ATUALMENTE == 'Sim':
            ULTIMO_SALARIO = st.number_input('Last Monthly Salary (R$)', min_value=0.0, value=5400.0, step=100.0)
        else:
            ULTIMO_SALARIO = 0.0

        TEMPO_ULTIMO_EMPREGO_MESES = st.slider('Months at Last Job', 0, 240, 5)
        QT_CARROS_input = st.number_input('Number of Cars', min_value=0, value=1)
        VALOR_TABELA_CARROS = st.slider('Total Value of Cars (R$)', 0, 200000, 45000, step=5000)
        
        OUTRA_RENDA = st.radio('Has Other Income?', ['Sim', 'Não'], index=1, horizontal=True)
        if OUTRA_RENDA == 'Sim':
            OUTRA_RENDA_VALOR = st.number_input('Other Income Amount (R$)', min_value=0.0, value=2000.0, step=100.0)
        else:
            OUTRA_RENDA_VALOR = 0.0

if st.button("Analyze Creditworthiness", type="primary"):
    # ------------------- Montar dados do input ------------------- #
    novos_dados_dict = {
        'UF': uf_map[UF], 'ESCOLARIDADE': escolaridade_map[ESCOLARIDADE], 'ESTADO_CIVIL': estado_civil_map[ESTADO_CIVIL], 'QT_FILHOS': int(QT_FILHOS),
        'CASA_PROPRIA': 1 if CASA_PROPRIA == 'Sim' else 0, 'QT_IMOVEIS': int(QT_IMOVEIS), 'VL_IMOVEIS': float(VL_IMOVEIS),
        'OUTRA_RENDA': 1 if OUTRA_RENDA == 'Sim' else 0, 'OUTRA_RENDA_VALOR': float(OUTRA_RENDA_VALOR), 'TEMPO_ULTIMO_EMPREGO_MESES': int(TEMPO_ULTIMO_EMPREGO_MESES),
        'TRABALHANDO_ATUALMENTE': 1 if TRABALHANDO_ATUALMENTE == 'Sim' else 0, 'ULTIMO_SALARIO': float(ULTIMO_SALARIO), 'QT_CARROS': int(QT_CARROS_input),
        'VALOR_TABELA_CARROS': float(VALOR_TABELA_CARROS), 'FAIXA_ETARIA': faixa_etaria_map[FAIXA_ETARIA]
    }

    X_input_df = pd.DataFrame([novos_dados_dict.values()], columns=feature_names)
    
    try:
        X_input_scaled = scaler.transform(X_input_df)
        X_input_scaled_df = pd.DataFrame(X_input_scaled, columns=feature_names)
        
        y_pred = lr_model.predict(X_input_scaled)[0]
        proba = lr_model.predict_proba(X_input_scaled)[0][1]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()

    # --- Container de Resultados para facilitar o Print Screen --- #
    with st.container(border=True):
        resultado_texto_en = 'Approved' if int(y_pred) == 1 else 'Declined'
        cor = 'green' if int(y_pred) == 1 else 'red'
        
        st.subheader("Prediction Result")
        st.markdown(f"### Result: <span style='color:{cor};'>{resultado_texto_en}</span>", unsafe_allow_html=True)
        st.write(f"Probability of Approval: **{proba:.2%}**")
        st.divider()

        # Acumuladores de explicações para o PDF
        shap_reasons_for_pdf = []
        lime_reasons_for_pdf = []
        llm_feedback_for_pdf = ""

        # ------------------- SHAP -------------------
        st.subheader("SHAP Explanation (Feature Impact)")
        fig_waterfall, ax = plt.subplots()
        try:
            explainer = shap.TreeExplainer(lr_model)
            sv_scaled = explainer(X_input_scaled_df)

            sv_plot = shap.Explanation(
                values=sv_scaled.values[0],
                base_values=sv_scaled.base_values[0],
                data=X_input_df.iloc[0].values,
                feature_names=feature_names
            )
            
            shap.plots.waterfall(sv_plot, show=False, max_display=10)
            st.pyplot(fig_waterfall)

            # Extrair razões SHAP
            contribs = sv_scaled.values[0]
            if y_pred == 0:
                idx = np.argsort(contribs)[:3] # Fatores negativos
                st.write("**Top 3 factors that contributed to the decline:**")
            else:
                idx = np.argsort(contribs)[-3:][::-1] # Fatores positivos
                st.write("**Top 3 factors that contributed to the approval:**")
            
            for j in idx:
                feature = feature_names[j]
                val = X_input_df.iloc[0, j]
                contrib = contribs[j]
                val_str = format_currency(val) if feature in ['VL_IMOVEIS', 'ULTIMO_SALARIO', 'VALOR_TABELA_CARROS', 'OUTRA_RENDA_VALOR'] else str(val)
                reason = f"**{feature}**: A value of **{val_str}** had a SHAP contribution of **{contrib:.2f}**."
                st.markdown(f"- {reason}")
                shap_reasons_for_pdf.append(reason.replace("**", ""))

        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")
        finally:
             plt.close(fig_waterfall)
        st.divider()

        # ------------------- LIME -------------------
        st.subheader("LIME Explanation (Local Rules)")
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_df.values,
                feature_names=feature_names,
                class_names=['Declined', 'Approved'],
                mode='classification'
            )
            lime_exp = lime_explainer.explain_instance(
                X_input_df.values[0],
                lambda x: lr_model.predict_proba(scaler.transform(pd.DataFrame(x, columns=feature_names))),
                num_features=5
            )
            lime_features = lime_exp.as_list()
            
            feature_translations_en = {name: name.replace('_', ' ').title() for name in feature_names}
            
            st.write("**Top rules influencing the decision:**")
            for rule, contrib in lime_features:
                human_rule, input_val_str = humanize_lime_rule(rule, feature_translations_en, novos_dados_dict)
                if input_val_str:
                    reason = f"The rule '{human_rule}' was relevant. Your value is **{input_val_str}**."
                else:
                    reason = f"The rule '{human_rule}' was relevant."
                st.markdown(f"- {reason}")
                lime_reasons_for_pdf.append(reason.replace("**", ""))
        
        except Exception as e:
            st.warning(f"Could not generate LIME explanation: {e}")
        st.divider()

        # ------------------- Feedback do LLM -------------------
        st.subheader("Expert Feedback (Generated by LLM)")
        if client:
            resultado_texto_pt = 'Aprovado' if int(y_pred) == 1 else 'Recusado'
            exp_rec_shap = "\n".join(shap_reasons_for_pdf)
            exp_rec_lime = "\n".join(lime_reasons_for_pdf)

            prompt = f"""
            You are a Senior Data Scientist, an expert in explaining Machine Learning model results to clients in a clear, objective, and humane way.
            The credit analysis model predicted the result '{resultado_texto_en}' for a client.

            Here are the technical explanations of the factors that most influenced this decision:
            - **SHAP (Attribute Contributions):**
            {exp_rec_shap}
            - **LIME (Decision Rules):**
            {exp_rec_lime}

            Based on the SHAP and LIME information, create friendly feedback for the client, following the instructions below:

            1.  **Analysis of the Result:** In a friendly and empathetic tone, explain the main reasons that led to the decision. Mention the key factors from SHAP and the rules from LIME in bullet points. For LIME results, explain in natural language how the condition influenced the '{resultado_texto_en}' outcome. Format monetary values correctly (e.g., R$50,000.00).

            2.  **Recommendations (if the result is 'Declined')**: If the credit was declined, state this clearly and provide 2 or 3 practical, actionable tips on how the client can improve their profile to increase their chances of approval in the future. If approved, congratulate the client and reinforce the positive points.

            3.  **Structure:** Divide your response into sections like "Analysis Result" and "Recommendations".

            Be direct, empathetic, and constructive. Be concise and avoid technical jargon.
            """
            try:
                with st.spinner("Generating personalized feedback with AI..."):
                    resp = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a Senior Data Scientist and an expert in client communication."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1, max_tokens=500
                    )
                    feedback_content = resp.choices[0].message.content
                    st.markdown(feedback_content)
                    llm_feedback_for_pdf = feedback_content

            except Exception as e:
                st.error(f"Error generating feedback from OpenAI: {e}")
        else:
            st.info("LLM feedback is not available because the OpenAI API key is not configured.")

        # --- NOVA FUNCIONALIDADE: Botão de Download PDF --- #
        st.divider()
        pdf_bytes = create_pdf_report(
            result_text=resultado_texto_en,
            proba=proba,
            shap_fig=fig_waterfall,
            shap_reasons=shap_reasons_for_pdf,
            lime_reasons=lime_reasons_for_pdf,
            llm_feedback=llm_feedback_for_pdf
        )
        if pdf_bytes:
            st.download_button(
                label="📥 Download Results as PDF",
                data=pdf_bytes,
                file_name="credit_analysis_report.pdf",
                mime="application/pdf"
            )
