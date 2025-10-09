# credit_app.py
import os
import re
import base64
import io
import markdown
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
import lime.lime_tabular
from openai import OpenAI
from dotenv import load_dotenv
from xhtml2pdf import pisa
from PIL import Image

# ------------------ UTILITY FUNCTIONS ------------------ #

def format_currency(value):
    """Formats a number into Brazilian currency (R$ 1.234,56). Safe for None/NaN."""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return "R$ 0,00"
    s = f"R$ {v:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def humanize_lime_rule(rule, feature_translations, input_values):
    """Receives a LIME rule string and returns a more human-readable version."""
    clauses = re.split(r'\s+and\s+|\s*&\s*', rule, flags=re.IGNORECASE)
    human_clauses = []
    feature_name = None

    for c in clauses:
        c = c.strip().strip('()')
        match = re.search(r'([A-Za-z_][A-Za-z0-9_]*)', c)
        if match:
            feature_name = match.group(1)
            translated_feature = feature_translations.get(feature_name, feature_name)
            
            nums = re.findall(r'([-+]?\d*\.?\d+)', c)
            formatted_c = c
            for num in nums:
                try:
                    v = float(num)
                    formatted_num = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    formatted_c = formatted_c.replace(num, formatted_num, 1)
                except ValueError:
                    pass
            
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


# --- PDF REPORT GENERATION FUNCTION (FINAL VERSION) ---
def create_pdf_report(result_text, proba, shap_fig, shap_reasons, lime_reasons, llm_feedback, input_data_dict):
    """
    Generates a complete PDF report from the analysis results,
    including the applicant's input profile with enhanced formatting.
    """
    # 1. Convert SHAP figure to a base64 image
    buf = io.BytesIO()
    shap_fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
    shap_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # 2. Convert LLM feedback from Markdown to HTML
    llm_feedback_html = markdown.markdown(llm_feedback)

    # 3. Build HTML tables for the input data
    personal_info_html = ""
    for label, value in input_data_dict["Personal & Employment Info"].items():
        personal_info_html += f"<tr><td class='label'>{label}</td><td class='value'>{value}</td></tr>"
    
    assets_info_html = ""
    for label, value in input_data_dict["Assets"].items():
        assets_info_html += f"<tr><td class='label'>{label}</td><td class='value'>{value}</td></tr>"

    # 4. Assemble the complete HTML content for the report
    html = f"""
    <html>
    <head>
        <style>
            @page {{ size: a4 portrait; margin: 1.5cm; }}
            body {{ font-family: 'Helvetica', 'Arial', sans-serif; color: #333; line-height: 1.5; }}
            h1, h2, h3 {{ font-family: 'Georgia', serif; font-weight: normal; }}
            h1 {{ color: #003366; text-align: center; border-bottom: 2px solid #003366; padding-bottom: 10px; margin-bottom: 30px; }}
            h2 {{ color: #0055A4; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 25px; }}
            .profile-summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
            .profile-summary-table > tbody > tr > td {{ width: 50%; vertical-align: top; padding: 0 10px; }}
            .inner-table {{ width: 100%; border-collapse: collapse; }}
            .inner-table th {{ text-align: left; font-size: 1.2em; padding-bottom: 10px; color: #333; }}
            .inner-table td {{ padding: 8px 0; border-bottom: 1px solid #f0f0f0; }}
            .inner-table td.label {{ font-weight: bold; font-size: 1.1em; color: #333; }}
            .inner-table td.value {{ text-align: right; }}
            .result {{ font-size: 1.2em; font-weight: bold; padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center; color: white; background-color: {'#28a745' if result_text == 'Approved' else '#dc3545'}; }}
            .probability {{ text-align: center; font-size: 1.1em; margin-bottom: 20px; }}
            .explanation-section ul {{ list-style-type: none; padding-left: 0; }}
            .explanation-section li {{ background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px; margin-bottom: 8px; }}
            .shap-image {{ text-align: center; margin-top: 20px; }}
            .llm-feedback p, .llm-feedback li {{ margin-bottom: 10px; }}
            .llm-feedback ul {{ padding-left: 20px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Credit Analysis Report</h1>
        <h2>Applicant Profile Summary</h2>
        <table class="profile-summary-table">
            <tr>
                <td>
                    <table class="inner-table">
                        <tr><th colspan="2">Personal & Employment Info</th></tr>
                        {personal_info_html}
                    </table>
                </td>
                <td>
                    <table class="inner-table">
                        <tr><th colspan="2">Assets</th></tr>
                        {assets_info_html}
                    </table>
                </td>
            </tr>
        </table>
        <h2>Analysis Results</h2>
        <div class="result">{result_text}</div>
        <div class="probability">Approval Probability: <strong>{proba*100:.2f}%</strong></div>
        <h2>SHAP Explanation (Feature Impact)</h2>
        <div class="explanation-section"><ul>{''.join([f'<li>{reason}</li>' for reason in shap_reasons])}</ul></div>
        <div class="shap-image"><img src="data:image/png;base64,{shap_img_base64}" /></div>
        <h2>LIME Explanation (Local Rules)</h2>
        <div class="explanation-section"><ul>{''.join([f'<li>{reason}</li>' for reason in lime_reasons])}</ul></div>
        <h2>Expert Feedback (AI Generated)</h2>
        <div class="llm-feedback">{llm_feedback_html}</div>
    </body>
    </html>
    """
    
    # 5. Convert HTML to PDF
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html), dest=pdf_buffer)
    
    if pisa_status.err:
        return None
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ------------------ STREAMLIT APP CONFIG ------------------ #
st.set_page_config(page_title="XAI Credit Analysis", layout="wide")
st.title("Creditworthiness Prediction and Explainability (XAI)")
st.markdown("Enter the applicant's details below to get a credit prediction and an explanation of the result.")

# --- Data Mappings (Values in PT for the model) ---
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

# ------------------ LOAD MODELS & DATA ------------------ #
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('modelo_regressao.pkl')
    X_train_raw = joblib.load('X_train.pkl')
    if isinstance(X_train_raw, np.ndarray):
        X_train_df = pd.DataFrame(X_train_raw, columns=feature_names)
    else:
        X_train_df = X_train_raw[feature_names] if list(X_train_raw.columns) != feature_names else X_train_raw
except Exception as e:
    st.error(f"Error loading models/data: {e}")
    st.stop()

# ------------------ OPENAI CLIENT ------------------ #
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if not client:
    st.warning("⚠️ OpenAI API key not configured. LLM feedback will be unavailable.")

# ------------------ USER INTERFACE INPUTS ------------------ #
with st.expander("Enter Applicant's Profile Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Personal & Employment Info")
        UF = st.selectbox('State (UF)', ufs, index=0)
        ESCOLARIDADE = st.selectbox('Education Level', escolaridades, index=1)
        ESTADO_CIVIL = st.selectbox('Marital Status', estados_civis, index=0)
        QT_FILHOS = st.number_input('Number of Children', min_value=0, value=1)
        FAIXA_ETARIA = st.radio('Age Group', faixas_etarias, index=2, horizontal=True)
        TRABALHANDO_ATUALMENTE = st.radio('Currently Employed?', ['Sim', 'Não'], index=0, horizontal=True)
        ULTIMO_SALARIO = st.number_input('Last Monthly Salary (R$)', min_value=0.0, value=5400.0, step=100.0, disabled=(TRABALHANDO_ATUALMENTE == 'Não'))
        TEMPO_ULTIMO_EMPREGO_MESES = st.slider('Months at Last Job', 0, 240, 5)

    with col2:
        st.subheader("Assets")
        CASA_PROPRIA = st.radio('Owns a Home?', ['Sim', 'Não'], index=0, horizontal=True)
        QT_IMOVEIS = st.number_input('Number of Properties', min_value=0, value=1, disabled=(CASA_PROPRIA == 'Não'))
        VL_IMOVEIS = st.number_input('Total Value of Properties (R$)', min_value=0.0, value=100000.0, step=10000.0, disabled=(CASA_PROPRIA == 'Não'))
        
        QT_CARROS_input = st.number_input('Number of Cars', min_value=0, value=1)
        VALOR_TABELA_CARROS = st.slider(
            'Total Value of Cars (R$)', 0, 200000, 45000, step=5000,
            disabled=(QT_CARROS_input == 0)
        )

        OUTRA_RENDA = st.radio('Has Other Income?', ['Sim', 'Não'], index=1, horizontal=True)
        OUTRA_RENDA_VALOR = st.number_input('Other Income Amount (R$)', min_value=0.0, value=2000.0, step=100.0, disabled=(OUTRA_RENDA == 'Não'))

# ------------------ MAIN LOGIC & ANALYSIS ------------------ #
if st.button("Analyze Creditworthiness", type="primary"):
    # --- Prepare input data for the model ---
    novos_dados_dict = {
        'UF': uf_map[UF], 'ESCOLARIDADE': escolaridade_map[ESCOLARIDADE], 'ESTADO_CIVIL': estado_civil_map[ESTADO_CIVIL], 
        'QT_FILHOS': int(QT_FILHOS), 'CASA_PROPRIA': 1 if CASA_PROPRIA == 'Sim' else 0, 
        'QT_IMOVEIS': int(QT_IMOVEIS) if CASA_PROPRIA == 'Sim' else 0, 
        'VL_IMOVEIS': float(VL_IMOVEIS) if CASA_PROPRIA == 'Sim' else 0.0,
        'OUTRA_RENDA': 1 if OUTRA_RENDA == 'Sim' else 0, 
        'OUTRA_RENDA_VALOR': float(OUTRA_RENDA_VALOR) if OUTRA_RENDA == 'Sim' else 0.0,
        'TEMPO_ULTIMO_EMPREGO_MESES': int(TEMPO_ULTIMO_EMPREGO_MESES),
        'TRABALHANDO_ATUALMENTE': 1 if TRABALHANDO_ATUALMENTE == 'Sim' else 0, 
        'ULTIMO_SALARIO': float(ULTIMO_SALARIO) if TRABALHANDO_ATUALMENTE == 'Sim' else 0.0,
        'QT_CARROS': int(QT_CARROS_input), 
        'VALOR_TABELA_CARROS': float(VALOR_TABELA_CARROS) if QT_CARROS_input > 0 else 0.0,
        'FAIXA_ETARIA': faixa_etaria_map[FAIXA_ETARIA]
    }
    X_input_df = pd.DataFrame([novos_dados_dict.values()], columns=feature_names)
    
    # --- Make prediction ---
    try:
        X_input_scaled = scaler.transform(X_input_df)
        X_input_scaled_df = pd.DataFrame(X_input_scaled, columns=feature_names)
        y_pred = model.predict(X_input_scaled)[0]
        proba = model.predict_proba(X_input_scaled)[0][1]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()

    # --- Prepare input summary for the PDF report ---
    input_summary_for_pdf = {
        "Personal & Employment Info": {
            "State (UF)": UF, "Education Level": ESCOLARIDADE, "Marital Status": ESTADO_CIVIL,
            "Number of Children": int(QT_FILHOS), "Age Group": FAIXA_ETARIA, "Currently Employed?": TRABALHANDO_ATUALMENTE,
            "Last Monthly Salary (R$)": format_currency(ULTIMO_SALARIO), "Months at Last Job": int(TEMPO_ULTIMO_EMPREGO_MESES)
        },
        "Assets": {
            "Owns a Home?": CASA_PROPRIA, "Number of Properties": int(QT_IMOVEIS),
            "Total Value of Properties (R$)": format_currency(VL_IMOVEIS), "Number of Cars": int(QT_CARROS_input),
            "Total Value of Cars (R$)": format_currency(VALOR_TABELA_CARROS), "Has Other Income?": OUTRA_RENDA,
            "Other Income Amount (R$)": format_currency(OUTRA_RENDA_VALOR)
        }
    }
    if CASA_PROPRIA == 'Não':
        del input_summary_for_pdf["Assets"]["Number of Properties"], input_summary_for_pdf["Assets"]["Total Value of Properties (R$)"]
    if TRABALHANDO_ATUALMENTE == 'Não':
        del input_summary_for_pdf["Personal & Employment Info"]["Last Monthly Salary (R$)"]
    if OUTRA_RENDA == 'Não':
        del input_summary_for_pdf["Assets"]["Other Income Amount (R$)"]
    if QT_CARROS_input == 0:
        del input_summary_for_pdf["Assets"]["Total Value of Cars (R$)"]

    # --- Display results in a container ---
    with st.container(border=True):
        resultado_texto_en = 'Approved' if int(y_pred) == 1 else 'Declined'
        cor = 'green' if int(y_pred) == 1 else 'red'
        
        st.subheader("Prediction Result")
        st.markdown(f"### Result: <span style='color:{cor};'>{resultado_texto_en}</span>", unsafe_allow_html=True)
        st.write(f"Probability of Approval: **{proba*100:.2f}%**")
        st.divider()

        shap_reasons_for_pdf, lime_reasons_for_pdf, llm_feedback_for_pdf = [], [], ""
        
        # --- SHAP Explanation ---
        st.subheader("SHAP Explanation (Feature Impact)")
        # Criamos a figura e o eixo explicitamente
        fig_waterfall, ax = plt.subplots()
        try:
            explainer = shap.TreeExplainer(model)
            sv_scaled = explainer(X_input_scaled_df)
            sv_plot = shap.Explanation(values=sv_scaled.values[0], base_values=sv_scaled.base_values[0], data=X_input_df.iloc[0].values, feature_names=feature_names)
            
            # --- CORREÇÃO DEFINITIVA: Passa o eixo (ax) explicitamente para o SHAP ---
            shap.plots.waterfall(sv_plot, show=False, max_display=10, ax=ax)
            
            # Agora o Streamlit sabe exatamente qual figura exibir
            st.pyplot(fig_waterfall)
            
            contribs = sv_scaled.values[0]
            idx = np.argsort(contribs)[-3:][::-1] if y_pred == 1 else np.argsort(contribs)[:3]
            st.write(f"**Top 3 factors contributing to the {'approval' if y_pred == 1 else 'decline'}:**")
            for j in idx:
                feature, val, contrib = feature_names[j], X_input_df.iloc[0, j], contribs[j]
                val_str = format_currency(val) if 'VL_' in feature or 'SALARIO' in feature or 'VALOR' in feature else str(val)
                reason = f"**{feature}**: A value of **{val_str}** had a SHAP contribution of **{contrib:.2f}**."
                st.markdown(f"- {reason}")
                shap_reasons_for_pdf.append(reason.replace("**", ""))
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")
        st.divider()

        # --- LIME Explanation ---
        st.subheader("LIME Explanation (Local Rules)")
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train_df.values, feature_names=feature_names, class_names=['Declined', 'Approved'], mode='classification')
            lime_exp = lime_explainer.explain_instance(X_input_df.values[0], lambda x: model.predict_proba(scaler.transform(pd.DataFrame(x, columns=feature_names))), num_features=5)
            st.write("**Top rules influencing the decision:**")
            for rule, _ in lime_exp.as_list():
                human_rule, input_val_str = humanize_lime_rule(rule, {name: name.replace('_', ' ').title() for name in feature_names}, novos_dados_dict)
                reason = f"The rule '{human_rule}' was relevant." + (f" Your value is **{input_val_str}**." if input_val_str else "")
                st.markdown(f"- {reason}")
                lime_reasons_for_pdf.append(reason.replace("**", ""))
        except Exception as e:
            st.warning(f"Could not generate LIME explanation: {e}")
        st.divider()

        # --- LLM Feedback ---
        if client:
            st.subheader("Expert Feedback (Generated by AI)")
            prompt = f"""You are an expert credit analyst. The model predicted '{resultado_texto_en}'. Based on these SHAP and LIME reasons, provide a clear, empathetic summary for the client.
            SHAP: {shap_reasons_for_pdf}
            LIME: {lime_reasons_for_pdf}
            Structure your response with a 'Result Analysis' and 'Recommendations' section. If declined, offer 2-3 actionable tips. If approved, congratulate them. Use bullet points. Be concise. Format currency as R$ 1.234,56."""
            try:
                with st.spinner("Generating personalized feedback with AI..."):
                    resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=500)
                    feedback_content = resp.choices[0].message.content
                    st.markdown(feedback_content)
                    llm_feedback_for_pdf = feedback_content
            except Exception as e:
                st.error(f"Error generating feedback from OpenAI: {e}")
        
        # --- PDF Download Button ---
        pdf_bytes = create_pdf_report(
            result_text=resultado_texto_en, proba=proba, shap_fig=fig_waterfall,
            shap_reasons=shap_reasons_for_pdf, lime_reasons=lime_reasons_for_pdf,
            llm_feedback=llm_feedback_for_pdf, input_data_dict=input_summary_for_pdf
        )
        if pdf_bytes:
            st.download_button(
                label="📥 Download Results as PDF", data=pdf_bytes,
                file_name="credit_analysis_report.pdf", mime="application/pdf"
            )
        
        plt.close(fig_waterfall)
