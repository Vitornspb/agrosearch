# ============================================================
# AgroSearch - Motor de Busca Inteligente
# Desafio: Pipeline de Pré-processamento + Índice Invertido + TF-IDF
# Restrição: tudo implementado "do zero" (sem scikit-learn/TfidfVectorizer)
# ============================================================

import streamlit as st
import re
import unicodedata
import math
import pandas as pd
from collections import defaultdict

st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")

# ------------------------------------------------------------
# 0. BASE DE DOCUMENTOS (hardcode sugerido pelo desafio)
# ------------------------------------------------------------
DOCUMENTOS_PADRAO = [
    "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
]

STOPWORDS_PT = [
    "a", "o", "os", "as", "e", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "para", "por", "com", "um", "uma", "uns", "umas", "que", "é",
    "ao", "aos", "à", "às", "se", "sua", "seu", "suas", "seus",
]


# ------------------------------------------------------------
# 1. FASE 1 — PIPELINE DE PRÉ-PROCESSAMENTO (do zero)
# ------------------------------------------------------------
def tokenizar(texto: str):
    """Separa o texto em tokens (palavras), ignorando pontuação."""
    return re.findall(r"\b\w+\b", texto)


def normalizar(token: str):
    """Minúsculas + remoção de acentos."""
    token = token.lower()
    token = unicodedata.normalize("NFD", token).encode("ascii", "ignore").decode("utf-8")
    return token


def remover_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS_PT]


def stemming_simples(palavra: str):
    """Stemmer rudimentar: corta sufixos comuns do português."""
    sufixos = ["amente", "acao", "mente", "ados", "adas", "ando", "endo",
               "avel", "ivel", "oso", "osa", "s"]
    for suf in sorted(sufixos, key=len, reverse=True):
        if palavra.endswith(suf) and len(palavra) - len(suf) >= 3:
            return palavra[: -len(suf)]
    return palavra


def preprocessar(texto: str, usar_stopwords: bool, usar_stemming: bool):
    """Executa o pipeline completo e devolve a lista final de tokens."""
    tokens = tokenizar(texto)
    tokens = [normalizar(t) for t in tokens]
    if usar_stopwords:
        tokens = remover_stopwords(tokens)
    if usar_stemming:
        tokens = [stemming_simples(t) for t in tokens]
    return tokens


# ------------------------------------------------------------
# 2. FASE 2 — ÍNDICE DIRETO / ÍNDICE INVERTIDO (do zero)
# ------------------------------------------------------------
def construir_indices(documentos, usar_stopwords, usar_stemming):
    indice_direto = {}
    for i, doc in enumerate(documentos):
        doc_id = f"Doc{i + 1}"
        indice_direto[doc_id] = preprocessar(doc, usar_stopwords, usar_stemming)

    indice_invertido = defaultdict(list)
    for doc_id, termos in indice_direto.items():
        for termo in termos:
            if doc_id not in indice_invertido[termo]:
                indice_invertido[termo].append(doc_id)

    return indice_direto, dict(sorted(indice_invertido.items()))


# ------------------------------------------------------------
# 3. FASE 3 — TF-IDF e RANQUEAMENTO (do zero)
# ------------------------------------------------------------
def calcular_tf(termo, tokens_doc):
    if not tokens_doc:
        return 0.0
    return tokens_doc.count(termo) / len(tokens_doc)


def calcular_idf(termo, indice_direto):
    N = len(indice_direto)
    df_t = sum(1 for tokens in indice_direto.values() if termo in tokens)
    if df_t == 0:
        return 0.0
    return math.log(N / df_t)


def vetor_tfidf(tokens_doc, vocabulario, indice_direto):
    """Vetor TF-IDF de um documento (ou de uma query) sobre o vocabulário."""
    vetor = {}
    for termo in vocabulario:
        tf = calcular_tf(termo, tokens_doc)
        idf = calcular_idf(termo, indice_direto)
        vetor[termo] = tf * idf
    return vetor


def similaridade_cosseno(v1: dict, v2: dict):
    termos = set(v1) | set(v2)
    prod_escalar = sum(v1.get(t, 0) * v2.get(t, 0) for t in termos)
    norma1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    norma2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if norma1 == 0 or norma2 == 0:
        return 0.0
    return prod_escalar / (norma1 * norma2)


# ------------------------------------------------------------
# INTERFACE
# ------------------------------------------------------------
st.title("🌱 AgroSearch — Motor de Busca Inteligente")
st.caption("Protótipo do motor de busca textual da AgroTech Solutions. "
           "Índice invertido e TF-IDF implementados do zero (sem scikit-learn).")

# ---- Sidebar: controles do pipeline ----
st.sidebar.header("⚙️ Configuração do Pré-processamento")
usar_stopwords = st.sidebar.checkbox("Remover Stopwords", value=True)
usar_stemming = st.sidebar.checkbox("Aplicar Stemming", value=True)
usar_cosseno = st.sidebar.checkbox("🎁 Bônus: usar Similaridade de Cosseno na busca", value=False)

st.sidebar.header("📄 Base de Documentos")
docs_text = st.sidebar.text_area(
    "Um documento por linha (editável):",
    value="\n".join(DOCUMENTOS_PADRAO),
    height=200,
)
documentos = [d.strip() for d in docs_text.split("\n") if d.strip()]

# Recalcula os índices sempre que qualquer opção mudar (vocabulário dinâmico)
indice_direto, indice_invertido = construir_indices(documentos, usar_stopwords, usar_stemming)
vocabulario = sorted(indice_invertido.keys())

# ---- Fase 1: Pipeline de pré-processamento ----
st.header("1️⃣ Pipeline de Pré-processamento")
doc_exemplo = st.selectbox("Ver o pipeline aplicado a um documento:", documentos)

tokens_1 = tokenizar(doc_exemplo)
tokens_2 = [normalizar(t) for t in tokens_1]
tokens_3 = remover_stopwords(tokens_2) if usar_stopwords else tokens_2
tokens_4 = [stemming_simples(t) for t in tokens_3] if usar_stemming else tokens_3

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**1. Tokenização**")
    st.write(tokens_1)
with c2:
    st.markdown("**2. Normalização**")
    st.write(tokens_2)
with c3:
    st.markdown(f"**3. Stopwords** {'✅' if usar_stopwords else '⏭️ (desligado)'}")
    st.write(tokens_3)
with c4:
    st.markdown(f"**4. Stemming** {'✅' if usar_stemming else '⏭️ (desligado)'}")
    st.write(tokens_4)

st.info(f"📚 Tamanho do vocabulário final (com as opções atuais): **{len(vocabulario)} termos**")

# ---- Fase 2: Índice Invertido ----
st.header("2️⃣ Índice Invertido")
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Índice Direto (Doc → Termos)")
    st.json(indice_direto)
with col_b:
    st.subheader("Índice Invertido (Termo → Docs)")
    st.json(indice_invertido)

# ---- Fase 3: Busca com TF-IDF ----
st.header("3️⃣ Busca e Ranqueamento (TF-IDF)")
query = st.text_input("Digite sua consulta (query):", "irrigação soja")

if st.button("🔍 Buscar", type="primary"):
    tokens_query = preprocessar(query, usar_stopwords, usar_stemming)

    if not tokens_query:
        st.warning("A consulta ficou vazia após o pré-processamento. Tente outros termos.")
    else:
        resultados = []

        if usar_cosseno:
            # ---- Bônus: Similaridade de cosseno ----
            vetor_query = vetor_tfidf(tokens_query, vocabulario, indice_direto)
            for doc_id, tokens_doc in indice_direto.items():
                vetor_doc = vetor_tfidf(tokens_doc, vocabulario, indice_direto)
                score = similaridade_cosseno(vetor_query, vetor_doc)
                resultados.append({"Documento": doc_id, "Similaridade de Cosseno": round(score, 4)})
            df_resultado = pd.DataFrame(resultados).sort_values(
                "Similaridade de Cosseno", ascending=False
            ).reset_index(drop=True)
            coluna_score = "Similaridade de Cosseno"
        else:
            # ---- TF-IDF somado por termo da query ----
            for doc_id, tokens_doc in indice_direto.items():
                score_total = 0.0
                detalhes = {}
                for termo in tokens_query:
                    tf = calcular_tf(termo, tokens_doc)
                    idf = calcular_idf(termo, indice_direto)
                    tfidf = tf * idf
                    score_total += tfidf
                    detalhes[termo] = round(tfidf, 4)
                linha = {"Documento": doc_id}
                linha.update({f"TF-IDF('{t}')": v for t, v in detalhes.items()})
                linha["TF-IDF Acumulado"] = round(score_total, 4)
                resultados.append(linha)
            df_resultado = pd.DataFrame(resultados).sort_values(
                "TF-IDF Acumulado", ascending=False
            ).reset_index(drop=True)
            coluna_score = "TF-IDF Acumulado"

        def destaca_vencedor(row):
            cor = "background-color: #d4f7d4" if row.name == 0 else ""
            return [cor] * len(row)

        st.dataframe(
            df_resultado.style.apply(destaca_vencedor, axis=1),
            use_container_width=True,
        )

        vencedor = df_resultado.iloc[0]
        st.success(
            f"🏆 Documento mais relevante: **{vencedor['Documento']}** "
            f"({coluna_score} = {vencedor[coluna_score]})\n\n"
            f"> {documentos[int(vencedor['Documento'].replace('Doc', '')) - 1]}"
        )
