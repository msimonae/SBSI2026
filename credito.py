# credito.py
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

# ------------------ Utilitários ------------------ #
def format_currency(value):
    """Formata número em R$ 1.234.567,89. Seguro contra None/NaN."""
    try:
        v = float(value)
    except Exception:
        return "R$ 0,00"
    s = f"R$ {v:,.2f}"
    # converte estilo en -> pt (ponto para milhar, vírgula para decimal)
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def format_shap_contrib(value):
    """Formata contribuição SHAP como número com ponto decimal (ex: -3.24)."""
    try:
        return f"{float(value):.2f}"
    except Exception:
        return f"{value}"

def _format_number_in_rule(num_str, is_money):
    try:
        v = float(num_str)
        if is_money:
            return format_currency(v)
        # se não for monetário, mantemos número com no máximo 2 casas decimais
        return f"{v:.2f}" if (abs(v) < 1000 and (v % 1) != 0) else str(int(v)) if v.is_integer() else f"{v:.2f}"
    except:
        return num_str


def humanize_lime_rule(rule, feature_translations, input_values):
    """
    Recebe uma string de regra do LIME (ex.: 'ULTIMO_SALARIO <= 3900.0')
    e retorna texto humanizado com valores formatados.
    """
    # traduz nomes de features
    # quebra conjunções (LIME costuma retornar expressões simples, mas tratamos & e " and ")
    clauses = re.split(r'\s+and\s+|\s*&\s*', rule, flags=re.IGNORECASE)
    human_clauses = []

    for c in clauses:
        c = c.strip().strip('()')
        # padrões possíveis:
        # 1) a < feature <= b  (ex: 0.0 < VALOR <= 185000.0)
        m_range = re.match(r'([-+]?\d*\.?\d+)\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*([-+]?\d*\.?\d+)', c)
        m_range2 = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s*>\s*([-+]?\d*\.?\d+)\s*and\s*([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*([-+]?\d*\.?\d+)', c, flags=re.IGNORECASE)
        m_le = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s*<=\s*([-+]?\d*\.?\d+)', c)
        m_ge = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*([-+]?\d*\.?\d+)', c)
        m_gt = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s*>\s*([-+]?\d*\.?\d+)', c)
        m_eq = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s*==?\s*["\']?([^"\']+)["\']?', c)

        # detecta feature para decidir formato monetário
        feature_name_search = re.search(r'([A-Za-z_][A-Za-z0-9_]*)', c)
        feature_name = feature_name_search.group(1) if feature_name_search else None
        is_money = feature_name in ['VL_IMOVEIS', 'VALOR_TABELA_CARROS', 'ULTIMO_SALARIO', 'OUTRA_RENDA_VALOR']

        translated = feature_translations.get(feature_name, feature_name if feature_name else c)

        if m_range:
            a, f, b = m_range.group(1), m_range.group(2), m_range.group(3)
            human_clauses.append(f"{translated} entre {_format_number_in_rule(a, is_money)} e {_format_number_in_rule(b, is_money)}")
        elif m_range2:
            # fallback raro - compõe manualmente
            _, a, _, b = m_range2.groups()
            human_clauses.append(f"{translated} entre {_format_number_in_rule(a, is_money)} e {_format_number_in_rule(b, is_money)}")
        elif m_le:
            f, val = m_le.group(1), m_le.group(2)
            human_clauses.append(f"{translated} igual ou menor que {_format_number_in_rule(val, is_money)}")
        elif m_ge:
            f, val = m_ge.group(1), m_ge.group(2)
            human_clauses.append(f"{translated} igual ou maior que {_format_number_in_rule(val, is_money)}")
        elif m_gt:
            f, val = m_gt.group(1), m_gt.group(2)
            human_clauses.append(f"{translated} maior que {_format_number_in_rule(val, is_money)}")
        elif m_eq:
            f, val = m_eq.group(1), m_eq.group(2)
            human_clauses.append(f"{translated} igual a {val}")
        else:
            # fallback: substitui números por formatados se existirem
            nums = re.findall(r'[-+]?\d*\.?\d+', c)
            c_fmt = c
            for n in nums:
                c_fmt = c_fmt.replace(n, _format_number_in_rule(n, is_money))
            human_clauses.append(c_fmt)

    humanized = " e ".join(human_clauses)
    # adiciona info do valor de entrada do usuário se disponível
    input_value = input_values.get(feature_name, None) if feature_name else None
    if input_value is not None:
        if feature_name in ['VL_IMOVEIS', 'VALOR_TABELA_CARROS', 'ULTIMO_SALARIO', 'OUTRA_RENDA_VALOR']:
            input_str = format_currency(input_value)
        else:
            input_str = str(input_value)
        return humanized, input_str
    else:
        return humanized, None


# ------------------ Config UI e mapeamentos ------------------ #
st.set_page_config(page_title="Crédito com XAI", layout="wide")
st.title("Previsão de Crédito e Explicabilidade (XAI)")

# Mapeamentos (exemplo)
ufs = ['SP', 'MG', 'SC', 'PR', 'RJ']
escolaridades = ['Superior Cursando', 'Superior Completo', 'Segundo Grau Completo']
estados_civis = ['Solteiro', 'Casado', 'Divorciado']
faixas_etarias = ['18-25', '26-35', '36-45', '46-60', 'Acima de 60']

uf_map = {label: i for i, label in enumerate(ufs)}
escolaridade_map = {label: i for i, label in enumerate(escolaridades)}
estado_civil_map = {label: i for i, label in enumerate(estados_civis)}
faixa_etaria_map = {label: i for i, label in enumerate(faixas_etarias)}

feature_names = [
    'UF', 'ESCOLARIDADE', 'ESTADO_CIVIL', 'QT_FILHOS', 'CASA_PROPRIA',
    'QT_IMOVEIS', 'VL_IMOVEIS', 'OUTRA_RENDA', 'OUTRA_RENDA_VALOR',
    'TEMPO_ULTIMO_EMPREGO_MESES', 'TRABALHANDO_ATUALMENTE', 'ULTIMO_SALARIO',
    'QT_CARROS', 'VALOR_TABELA_CARROS', 'FAIXA_ETARIA'
]

# ------------------ OPENAI Client ------------------ #
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.warning("⚠️ OPENAI_API_KEY não configurada. O feedback do LLM não estará disponível.")

# ------------------ Carregar modelos/dados (joblib) ------------------ #
try:
    scaler = joblib.load('scaler.pkl')
    lr_model = joblib.load('modelo_regressao.pkl')
    X_train_scaled = joblib.load('X_train_scaled.pkl')  # escalado
    X_train_raw = joblib.load('X_train.pkl')           # raw (para LIME/Anchor)
    # garante dataframe com colunas corretas
    if isinstance(X_train_raw, np.ndarray):
        X_train_df = pd.DataFrame(X_train_raw, columns=feature_names)
    else:
        # se falta alguma coluna, seleciona na ordem certa
        if list(X_train_raw.columns) != feature_names:
            X_train_df = X_train_raw[feature_names]
        else:
            X_train_df = X_train_raw
except Exception as e:
    st.error(f"Erro ao carregar modelos/dados: {e}")
    st.stop()

# ------------------ Inputs UI ------------------ #
col1, col2, col3 = st.columns(3)
with col1:
    UF = st.selectbox('UF', ufs, index=0)
    ESCOLARIDADE = st.selectbox('Escolaridade', escolaridades, index=1)
    ESTADO_CIVIL = st.selectbox('Estado Civil', estados_civis, index=0)
    QT_FILHOS = st.number_input('Qtd. Filhos', min_value=0, value=1)
    CASA_PROPRIA = st.radio('Casa Própria?', ['Sim', 'Não'], index=0)
with col2:
    if CASA_PROPRIA == 'Sim':
        QT_IMOVEIS = st.number_input('Qtd. Imóveis', min_value=0, value=1)
        VL_IMOVEIS_str = st.text_input('Valor dos Imóveis (R$)', value="100000")
        try:
            VL_IMOVEIS = float(VL_IMOVEIS_str.replace("R$", "").replace(".", "").replace(",", "."))
        except:
            VL_IMOVEIS = 0.0
    else:
        QT_IMOVEIS = 0
        VL_IMOVEIS = 0.0

    OUTRA_RENDA = st.radio('Outra renda?', ['Sim', 'Não'], index=1)
    OUTRA_RENDA_VALOR = 0.0
    if OUTRA_RENDA == 'Sim':
        OUTRA_RENDA_VALOR_str = st.text_input('Valor Outra Renda (R$)', value="2000")
        try:
            OUTRA_RENDA_VALOR = float(OUTRA_RENDA_VALOR_str.replace("R$", "").replace(".", "").replace(",", "."))
        except:
            OUTRA_RENDA_VALOR = 0.0
    TEMPO_ULTIMO_EMPREGO_MESES = st.slider('Tempo Últ. Emprego (meses)', 0, 240, 5)

with col3:
    TRABALHANDO_ATUALMENTE = st.radio('Trabalhando atualmente?', ['Sim', 'Não'], index=0)
    trabalhando_flag = 1 if TRABALHANDO_ATUALMENTE == 'Sim' else 0

    if TRABALHANDO_ATUALMENTE == 'Sim':
        ULTIMO_SALARIO_str = st.text_input('Último Salário (R$)', value="5400")
        try:
            ULTIMO_SALARIO = float(ULTIMO_SALARIO_str.replace("R$", "").replace(".", "").replace(",", "."))
        except:
            ULTIMO_SALARIO = 0.0
    else:
        ULTIMO_SALARIO = 0.0

    QT_CARROS_input = st.number_input('Qtd. Carros', min_value=0, value=1)
    VALOR_TABELA_CARROS = st.slider('Valor Tabela Carros (R$)', 0, 200000, 45000, step=5000)
    FAIXA_ETARIA = st.radio('Faixa Etária', faixas_etarias, index=2)

if st.button("Verificar Crédito"):
    # ------------------- Montar dados do input ------------------- #
    novos_dados_dict = {
        'UF': uf_map[UF], 'ESCOLARIDADE': escolaridade_map[ESCOLARIDADE], 'ESTADO_CIVIL': estado_civil_map[ESTADO_CIVIL], 'QT_FILHOS': int(QT_FILHOS),
        'CASA_PROPRIA': 1 if CASA_PROPRIA == 'Sim' else 0, 'QT_IMOVEIS': int(QT_IMOVEIS), 'VL_IMOVEIS': float(VL_IMOVEIS),
        'OUTRA_RENDA': 1 if OUTRA_RENDA == 'Sim' else 0, 'OUTRA_RENDA_VALOR': float(OUTRA_RENDA_VALOR), 'TEMPO_ULTIMO_EMPREGO_MESES': int(TEMPO_ULTIMO_EMPREGO_MESES),
        'TRABALHANDO_ATUALMENTE': 1 if TRABALHANDO_ATUALMENTE == 'Sim' else 0, 'ULTIMO_SALARIO': float(ULTIMO_SALARIO), 'QT_CARROS': int(QT_CARROS_input),
        'VALOR_TABELA_CARROS': float(VALOR_TABELA_CARROS), 'FAIXA_ETARIA': faixa_etaria_map[FAIXA_ETARIA]
    }

    novos_dados = list(novos_dados_dict.values())
    X_input_df = pd.DataFrame([novos_dados], columns=feature_names)
    # Escala para o modelo
    try:
        X_input_scaled = scaler.transform(X_input_df)
    except Exception as e:
        st.error(f"Erro ao transformar dados com scaler: {e}")
        st.stop()
    X_input_scaled_df = pd.DataFrame(X_input_scaled, columns=feature_names)

    # Predição
    try:
        y_pred = lr_model.predict(X_input_scaled)[0]
        proba = getattr(lr_model, "predict_proba", lambda x: np.array([[1, 0]]))(X_input_scaled)[0][1]
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        st.stop()

    resultado_texto = 'Aprovado' if int(y_pred) == 1 else 'Recusado'
    cor = 'green' if int(y_pred) == 1 else 'red'
    st.markdown(f"### Resultado: <span style='color:{cor}; font-weight:700'>{resultado_texto}</span>", unsafe_allow_html=True)
    st.write(f"Probabilidade de Aprovação: **{proba:.2%}**")

    # Acumuladores de explicações
    exp_rec_shap = ""
    exp_rec_lime = ""
    exp_rec_anchor = ""

        # ------------------- SHAP -------------------
    try:
        st.markdown("**Explicação com SHAP (Impacto das Features na Predição Atual):**")
        explainer = shap.TreeExplainer(lr_model)
        sv_scaled = explainer(X_input_scaled_df)

        sv_plot = shap.Explanation(
            values=sv_scaled.values[0],
            base_values=sv_scaled.base_values[0],
            data=X_input_df.iloc[0].values,
            feature_names=feature_names
        )

        fig_waterfall = plt.figure()
        shap.plots.waterfall(sv_plot, show=False, max_display=10)
        st.pyplot(fig_waterfall)
        plt.close(fig_waterfall)

        contribs = sv_scaled.values[0]
        if y_pred == 0:
            idx = np.argsort(contribs)[:3]
            st.write("**SHAP - Principais fatores que influenciaram a recusa:**")
        else:
            idx = np.argsort(contribs)[-3:]
            st.write("**SHAP - Principais fatores que influenciaram a aprovação:**")

        razoes_shap_list = []
        for j in idx:
            feature_name = feature_names[j]
            contrib = contribs[j]
            val = X_input_df.iloc[0, j]
            if feature_name in ['VL_IMOVEIS', 'ULTIMO_SALARIO', 'VALOR_TABELA_CARROS', 'OUTRA_RENDA_VALOR']:
                val_str = format_currency(val)
                razoes_shap_list.append(f"{feature_name}: contribuição de {contrib:.2f}, com um valor de {val_str}.")
            else:
                razoes_shap_list.append(f"{feature_name}: contribuição de {contrib:.2f}, com um valor de {val}.")
            
        exp_rec_shap = "\n".join(razoes_shap_list)

        for r in razoes_shap_list:
            st.markdown(f"- {r}")

    except Exception as e:
        st.warning(f"Não foi possível gerar SHAP: {e}")

    
    # # ------------------- SHAP (explicador robusto) ------------------- #
    # try:
    #     st.markdown("**Explicação com SHAP (Impacto das Features na probabilidade de aprovação):**")
    #     # Explicador que foca na probabilidade da classe 1 (aprovado)
    #     # Criação do explainer
    #     explainer = shap.Explainer(lr_model, X_train_df, feature_names=feature_names)
    #     #explainer = shap.Explainer(lambda x: lr_model.predict_proba(x)[:, 1], X_train_scaled)
    #     #shap_values = explainer(X_input_scaled)
    #     shap_values = explainer(X_input_df)
    #     # waterfall plot
    #     fig = plt.figure()
    #     # Gráfico Waterfall (com nomes corretos)
    #     shap.plots.waterfall(shap_values[0], show=True)
    #     #shap.plots.waterfall(shap_values[0], show=False, max_display=10)
    #     st.pyplot(fig)
    #     plt.close(fig)

    #     contribs = shap_values[0].values  # array (n_features,)
    #     # seleciona top 3 que favorecem ou top 3 que prejudicam a decisão
    #     if int(y_pred) == 0:
    #         # para recusa: pega menores (mais negativos)
    #         idx = np.argsort(contribs)[:3]
    #         st.write("**SHAP - Principais fatores que influenciaram a recusa:**")
    #     else:
    #         idx = np.argsort(contribs)[-3:]
    #         st.write("**SHAP - Principais fatores que influenciaram a aprovação:**")

    #     razoes_shap_list = []
    #     for j in idx:
    #         feature_name = feature_names[j]
    #         contrib = contribs[j]
    #         val = X_input_df.iloc[0, j]
    #         # valor de entrada formatado quando for monetário
    #         if feature_name in ['VL_IMOVEIS', 'ULTIMO_SALARIO', 'VALOR_TABELA_CARROS', 'OUTRA_RENDA_VALOR']:
    #             val_str = format_currency(val)
    #         else:
    #             val_str = str(val)
    #         razoes_shap_list.append({
    #             "feature": feature_name,
    #             "contrib": format_shap_contrib(contrib),
    #             "valor": val_str
    #         })
    #         st.markdown(f"- **{feature_name}**: contribuição **{format_shap_contrib(contrib)}**, valor: **{val_str}**")

    #     exp_rec_shap = "\n".join([f"{r['feature']}: contribuição {r['contrib']}, valor {r['valor']}" for r in razoes_shap_list])
    # except Exception as e:
    #     st.warning(f"Não foi possível gerar SHAP: {e}")

    # ------------------- LIME ------------------- #
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_df.values,
            feature_names=feature_names,
            class_names=['Recusado', 'Aprovado'],
            mode='classification'
        )
        lime_exp = lime_explainer.explain_instance(
            X_input_df.values[0],
            lambda x: lr_model.predict_proba(scaler.transform(pd.DataFrame(x, columns=feature_names))),
            num_features=5
        )
        lime_features = lime_exp.as_list()  # lista de (rule, contrib)

        # Exibição BRUTA no Streamlit
        st.markdown("**LIME – Principais fatores (regras brutas):**")
        for rule, contrib in lime_features:
            st.write(f"- Regra LIME: `{rule}`, contribuição: {contrib:.4f}")

        # Versão humanizada (apenas para o LLM)
        feature_translations = {
            'VL_IMOVEIS': 'o valor dos seus imóveis',
            'VALOR_TABELA_CARROS': 'o valor de tabela dos seus carros',
            'TEMPO_ULTIMO_EMPREGO_MESES': 'o seu tempo de último emprego (meses)',
            'ULTIMO_SALARIO': 'o seu último salário',
            'QT_CARROS': 'a quantidade de carros',
        }

        exp_rec_lime_human = []
        for rule, contrib in lime_features:
            # impacto
            if int(y_pred) == 0:
                impact = "negativo"
            else:
                impact = "positivo" if contrib > 0 else "negativo"

            human_rule, input_val_str = humanize_lime_rule(rule, feature_translations, novos_dados_dict)
            if input_val_str:
                exp_rec_lime_human.append(f"Regra LIME: '{human_rule}'. Seu valor: {input_val_str}. Impacto: {impact}.")
            else:
                exp_rec_lime_human.append(f"Regra LIME: '{human_rule}'. Impacto: {impact}.")

        # Para o LLM
        exp_rec_lime = "\n".join(exp_rec_lime_human)

    except Exception as e:
        st.warning(f"Não foi possível gerar LIME: {e}")

    
    # # ------------------- LIME ------------------- #
    # try:
    #     lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    #         training_data=X_train_df.values,
    #         feature_names=feature_names,
    #         class_names=['Recusado', 'Aprovado'],
    #         mode='classification'
    #     )
    #     lime_exp = lime_explainer.explain_instance(
    #         X_input_df.values[0],
    #         lambda x: lr_model.predict_proba(scaler.transform(pd.DataFrame(x, columns=feature_names))),
    #         num_features=5
    #     )
    #     lime_features = lime_exp.as_list()  # lista de (rule, contrib)

    #     feature_translations = {
    #         'VL_IMOVEIS': 'o valor dos seus imóveis',
    #         'VALOR_TABELA_CARROS': 'o valor de tabela dos seus carros',
    #         'TEMPO_ULTIMO_EMPREGO_MESES': 'o seu tempo de último emprego (meses)',
    #         'ULTIMO_SALARIO': 'o seu último salário',
    #         'QT_CARROS': 'a quantidade de carros',
    #     }

    #     exp_rec_lime_list = []
    #     for rule, contrib in lime_features:
    #         # decide impacto relative ao y_pred e ao sinal da contribuição
    #         if int(y_pred) == 0:
    #             impact = "negativo"
    #         else:
    #             impact = "positivo" if contrib > 0 else "negativo"

    #         human_rule, input_val_str = humanize_lime_rule(rule, feature_translations, novos_dados_dict)
    #         if input_val_str:
    #             exp_rec_lime_list.append(f"Regra LIME: '{human_rule}'. Seu valor: {input_val_str}. Impacto: {impact}.")
    #         else:
    #             exp_rec_lime_list.append(f"Regra LIME: '{human_rule}'. Impacto: {impact}.")

    #     exp_rec_lime = "\n".join(exp_rec_lime_list)
    #     st.markdown("**LIME – Principais fatores (humanizados):**")
    #     for item in exp_rec_lime_list:
    #         st.write(f"- {item}")

    # except Exception as e:
    #     st.warning(f"Não foi possível gerar LIME: {e}")

    # ------------------- ELI5 ------------------- #
    try:
        eli5_expl = eli5.explain_prediction(lr_model, X_input_df.iloc[0], feature_names=feature_names)
        html_eli5 = format_as_html(eli5_expl)
        # Exibe o HTML (se quiser ativar)
        # st.components.v1.html(html_eli5, height=420, scrolling=True)
    except Exception as e:
        st.warning(f"Não foi possível gerar ELI5: {e}")

    # ------------------- Anchor -------------------
    try:
        #st.markdown("**Explicação com Anchor (Regras Mínimas):**")
        def predict_fn_anchor(arr2d):
            df = pd.DataFrame(arr2d, columns=feature_names)
            scaled = scaler.transform(df)
            return lr_model.predict(scaled)

        anchor_explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=['Recusado', 'Aprovado'],
            feature_names=feature_names,
            train_data=X_train_df.values
        )
        anchor_exp = anchor_explainer.explain_instance(
            X_input_df.values[0], predict_fn_anchor, threshold=0.95
        )
        
        rule = " E ".join(anchor_exp.names())
        #st.write(f"**Anchor – Regra que ancora a predição:** Se *{rule}*, então o resultado é **{resultado_texto}**.")
        #st.write(f"Precisão da regra: {anchor_exp.precision():.2f} | Cobertura da regra: {anchor_exp.coverage():.2f}")
        exp_rec_anchor = f"Regra Anchor: {rule}"

    except Exception as e:
        st.warning(f"Não foi possível gerar a explicação Anchor: {e}")

    # ------------------- Feedback do LLM -------------------
    if client:
        prompt = f"""
Você é um Cientista de Dados Sênior, especialista em explicar os resultados de modelos de Machine Learning para clientes de forma clara, objetiva e humana.
O modelo de análise de crédito previu o resultado '{resultado_texto}' para um cliente.

Aqui estão as explicações técnicas sobre os fatores que mais influenciaram essa decisão:
- **SHAP (Contribuição dos atributos):**
{exp_rec_shap}
- **LIME (Regras de decisão):**
{exp_rec_lime}

Com base nas informações do **SHAP** e **LIME**, crie um feedback amigável para o cliente, seguindo as instruções abaixo:

1.  **Análise do Resultado:** De forma amigável e empática, explique todas as principais motivos que levaram à decisão. Mencione todos os fatores do SHAP e todas as regras do LIME e **liste em bullet points**. Para os resultados SHAP descrever apenas as variáveis importantes.Para todos os resultado do LIME, explique em linguagem natural como a condição do fator influenciaram o resultado '{resultado_texto}' e compare com o seu limite da regra do LIME {exp_rec_lime}. Formate valores monetários com R$ e use vírgulas e pontos decimais de forma correta (Exemplo: R$ 50.000,00).

2.  **Recomendações (se o resultado for 'Recusado')**: Se o crédito foi recusado, Diga que seu crédito foi recusado e forneça 2 ou 3 dicas práticas e acionáveis sobre como o cliente pode melhorar seu perfil para aumentar as chances de aprovação no futuro. Se foi aprovado, parabenize o cliente e apenas reforce os pontos positivos.

3.  **Estrutura:** Divida sua resposta em tópicos, como "Resultado da Análise" descrever se foi aprovado ou reprovado aqui e "Recomendações".

Seja direto, empático e construtivo. Evite qualquer tipo de concatenação de palavras. Formate valores monetários como R$ 1.234.567,89. Seja conciso, empático e evite jargões técnicos.
"""
        
#     # ------------------- Feedback do LLM (OpenAI) ------------------- #
#     if client:
#         prompt = f"""
# Você é um Cientista de Dados Sênior, especialista em explicar resultados de modelos de crédito para clientes de forma clara, empática e direta.
# Importante: As contribuições do SHAP (a seguir) estão em escala numérica do modelo e NÃO representam valores monetários. Os valores monetários aparecem entre parênteses ao lado de cada contribuição.

# Resultado previsto: '{resultado_texto}'

# SHAP (contribuições numéricas):
# {exp_rec_shap}

# LIME (regras humanizadas):
# {exp_rec_lime}

# Com base nisso, gere um texto amigável dividido em:
# 1) Análise do seu Perfil Financeiro — explique os principais motivos que levaram à decisão (liste bullet points com os fatores SHAP e as regras LIME, explicando em linguagem simples).
# 2) Recomendações / Pontos a Melhorar — se o resultado for 'Recusado', dê 2-3 dicas práticas e concretas.

# Formate valores monetários como R$ 1.234.567,89. Seja conciso, empático e evite jargões técnicos.
# """
        try:
            with st.spinner("Gerando feedback personalizado com o LLM..."):
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Você é um Cientista de Dados Sênior e especialista em comunicação com clientes."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                st.markdown("### 🔍 Feedback do Especialista (LLM)")
                feedback_content = resp.choices[0].message.content
                st.write(feedback_content)
        except Exception as e:
            st.error(f"Erro ao gerar feedback com a OpenAI: {e}")
    else:
        st.info("Feedback do LLM não gerado porque a chave OPENAI_API_KEY não está configurada.")
