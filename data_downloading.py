import os
import pickle
from sklearn.datasets import fetch_20newsgroups

DATA_DIR = "data"

def garantir_pasta_data():
    """Cria a pasta 'data' se ela ainda não existir."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Pasta '{DATA_DIR}/' criada com sucesso.")

def carregar_ou_baixar(nome_arquivo, funcao_download):
    """Função auxiliar que gerencia o cache local dos datasets."""
    garantir_pasta_data()
    caminho_arquivo = os.path.join(DATA_DIR, nome_arquivo)
    
    # Se o arquivo já existe localmente, carrega direto do disco
    if os.path.exists(caminho_arquivo):
        print(f"Carregando {nome_arquivo} do cache local...")
        with open(caminho_arquivo, 'rb') as f:
            return pickle.load(f)
            
    # Se não existe, faz o download e salva para a próxima vez
    print(f"{nome_arquivo} não encontrado localmente.")
    dados = funcao_download()
    
    print(f"Salvando {nome_arquivo} para cache futuro...")
    with open(caminho_arquivo, 'wb') as f:
        pickle.dump(dados, f)
        
    return dados

# --- Suas funções originais adaptadas para usar o cache ---

def download_data_completa():
    return carregar_ou_baixar(
        "newsgroups_completo.pkl", 
        lambda: fetch_20newsgroups()
    )

def download_data_subset_removida():
    return carregar_ou_baixar(
        "colecao_subset_removida.pkl", 
        lambda: fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    )
def download_data_subset_query():
    return carregar_ou_baixar(
        "colecao_subset_query.pkl", 
        lambda: fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    )
def download_data_set_categorias(categorias, type):

    return carregar_ou_baixar(
        "colecao_subset_removida.pkl",
        lambda: fetch_20newsgroups(
            subset=type,
            categories=categorias,
            remove=('headers', 'footers', 'quotes')
        )
    )