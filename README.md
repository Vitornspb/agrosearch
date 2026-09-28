# AgroSearch — Motor de Busca Inteligente para Documentos Agrícolas

**Laboratório Prático 04 — Desafio Integrador** (0.5 pontos extras na AV1)
**Disciplina:** Tendências em Ciência da Computação (UNIPÊ)
**Professor:** Me. Ricardo Roberto de Lima

## O que é

Um motor de busca de informação implementado do zero (sem bibliotecas prontas de IR, como `TfidfVectorizer`), com interface em [Streamlit](https://streamlit.io/):

- **Pré-processamento de texto:** tokenização por regex, normalização (minúsculas + remoção de acentos), remoção de stopwords em português e stemming simples (corte de sufixos) — cada etapa pode ser ligada/desligada pela barra lateral.
- **Índices:** índice direto e índice invertido, construídos manualmente sobre a base de 5 documentos agrícolas (editável na interface).
- **Ranqueamento:** cálculo de TF, IDF e TF-IDF do zero, com ranqueamento por soma de TF-IDF e, como bônus, por similaridade de cosseno.

## Estrutura do repositório

| Arquivo | Descrição |
|---|---|
| [`agrosearch_app.py`](agrosearch_app.py) | App Streamlit com o pipeline completo (pré-processamento → índice invertido → TF-IDF → cosseno) |
| [`relatorio_agrosearch.pdf`](relatorio_agrosearch.pdf) | Relatório técnico (2 páginas): contexto, arquitetura, fórmulas, validação e divisão de tarefas |

## Como rodar

```bash
pip install streamlit
streamlit run agrosearch_app.py
```

O app abre em <http://localhost:8501>. A base de documentos e as opções de pré-processamento (stopwords, stemming, cosseno) podem ser ajustadas pela barra lateral.

## Autoria

Equipe (dupla):

- Vitor Nóbrega de Souza
- Augusto Lustosa

Ver [`relatorio_agrosearch.pdf`](relatorio_agrosearch.pdf), seção de divisão de tarefas, para os detalhes de quem fez cada fase.

## Transparência

Este projeto foi desenvolvido com auxílio de ferramentas de Inteligência Artificial Generativa (Claude, da Anthropic) para geração e revisão de código, documentação técnica e organização do repositório.
