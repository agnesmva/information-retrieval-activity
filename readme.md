# Recuperação da Informação - Modelo Vetorial e BM25

Projeto prático desenvolvido para a disciplina **Recuperação da Informação** (2026.1)  
Ministrada pelo **Prof. Dr. Pedronette**  
Programa de Pós-Graduação em Computação – UNESP

## Descrição

Este projeto implementa dois modelos clássicos de recuperação da informação:

- **Modelo Vetorial (TF-IDF + cosseno)**
- **Modelo BM25 (Okapi BM25)**

O sistema utiliza um **corpus público** (ex.: coleção de documentos textuais) e permite que o usuário realize consultas (_queries_) para recuperar os documentos mais relevantes segundo cada modelo.

## Pré-requisitos

- [uv](https://docs.astral.sh/uv/) instalado no sistema

## Configuração do ambiente virtual

Recomenda-se criar um ambiente virtual antes de instalar as dependências.

```bash
uv venv
source .venv/bin/activate --linux/macOs
.venv\Scripts\activate    --PoweShell
```

## Instalação das dependências
```bash
uv pip install -r requirements.txt
```
