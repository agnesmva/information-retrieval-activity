import math
from collections import Counter

def build_vocabulary(docs):
    """
    Constrói o vocabulário do corpus e cria o mapeamento de índices.

    Parameters
    ----------
    docs : list[list[str]]
        Lista de documentos tokenizados.

    Returns
    -------
    vocabulary : list[str]
        Lista ordenada de termos únicos.
    term_to_idx : dict
        Mapeamento termo -> posição (índice) no vetor.
    """
    vocabulary = sorted(
        set(
            token
            for doc in docs
            for token in doc
        )
    )

    term_to_idx = {
        term: idx
        for idx, term in enumerate(vocabulary)
    }

    return vocabulary, term_to_idx


def calcular_frequencia_documento(docs, vocabulary, term_to_idx):
    """
    Calcula o vetor de Frequência do Documento (DF) alinhado ao vocabulário.
    
    Parâmetros:
    - docs: Lista de listas de strings (documentos tokenizados)
    - vocabulary: Lista ordenada de termos únicos
    - term_to_idx: Dicionário mapeando termo para seu índice no vetor
    
    Retorna:
    - df_vector: Lista (vetor) com a contagem de documentos de cada termo
    """
    df_vector = [0] * len(vocabulary)
    
    for doc in docs:
        unique_terms = set(doc)
        for term in unique_terms:
            if term in term_to_idx:
                idx = term_to_idx[term]
                df_vector[idx] += 1
                
    return df_vector


def calcular_idf(df_vector, total_documentos):
    """
    Calcula o vetor IDF (Inverse Document Frequency) com suavização.
    
    Parâmetros:
    - df_vector: Vetor contendo a frequência documental de cada termo
    - total_documentos: Número total de documentos no corpus (N)
    
    Retorna:
    - idf_vector: Lista com os pesos IDF calculados para cada posição do vocabulário
    """
    N = total_documentos
    idf_vector = []
    
    for df_t in df_vector:
        # Fórmula clássica com suavização (Smooth IDF) para evitar divisões por zero
        valor_idf = math.log((1 + N) / (1 + df_t)) + 1
        idf_vector.append(valor_idf)
        
    return idf_vector


def calcular_tf_idf_matriz(docs, vocabulary, term_to_idx, idf_vector):
    """
    Constrói as matrizes numéricas posicionais de TF e TF-IDF para o corpus.
    
    Parâmetros:
    - docs: Lista de listas de strings (documentos tokenizados)
    - vocabulary: Lista com os termos ordenados do vocabulário
    - term_to_idx: Dicionário de mapeamento termo -> índice
    - idf_vector: Vetor contendo os pesos IDF calculados
    
    Retorna:
    - tf_matrix: Lista de listas com as frequências brutas (TF)
    - tfidf_matrix: Lista de listas com os pesos finais (TF-IDF)
    """
    tf_matrix = []
    tfidf_matrix = []
    
    for doc in docs:
        counts = Counter(doc)
        
        # Cria vetores zerados do tamanho exato do vocabulário para o documento atual
        vetor_tf = [0.0] * len(vocabulary)
        vetor_tfidf = [0.0] * len(vocabulary)
        
        for term, freq in counts.items():
            if term in term_to_idx:
                idx = term_to_idx[term]
                
                # 1. Armazena o TF bruto
                vetor_tf[idx] = float(freq)
                
                # 2. Calcula o TF-IDF baseado na posição correta do vetor IDF
                vetor_tfidf[idx] = float(freq) * idf_vector[idx]
                
        tf_matrix.append(vetor_tf)
        tfidf_matrix.append(vetor_tfidf)
        
    return tf_matrix, tfidf_matrix

def calcular_tf_idf_query(query_tokens, vocabulary, term_to_idx, idf_vector):
    """
    Transforma uma query tokenizada em um vetor numérico TF-IDF alinhado ao vocabulário.
    
    Parâmetros:
    - query_tokens: Lista de strings (ex: ['python', 'code', 'error'])
    - vocabulary: Lista com os termos ordenados do vocabulário do corpus
    - term_to_idx: Dicionário de mapeamento termo -> índice
    - idf_vector: Vetor contendo os pesos IDF calculados no corpus de treino
    
    Retorna:
    - query_tfidf: Lista contendo o vetor TF-IDF da query
    """
    # Conta a frequência dos termos na query
    counts = Counter(query_tokens)
    
    # Cria o vetor zerado do tamanho exato do vocabulário
    query_tfidf = [0.0] * len(vocabulary)
    
    for term, freq in counts.items():
        # Importante: ignoramos palavras da query que NÃO existem no vocabulário do corpus
        if term in term_to_idx:
            idx = term_to_idx[term]
            
            # Calcula o TF-IDF da query (TF bruto da query * IDF histórico do corpus)
            query_tfidf[idx] = float(freq) * idf_vector[idx]
            
    return query_tfidf