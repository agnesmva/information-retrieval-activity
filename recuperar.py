import math
from sklearn.metrics.pairwise import cosine_similarity

def normalize(vector):

    norm = math.sqrt(
        sum(x*x for x in vector)
    )

    if norm == 0:
        return vector

    return [
        x / norm
        for x in vector
    ]

def rank_documents(query_vector, tfidf_matrix):
    # 1. Passa a query como 2D e compara com a matriz inteira de uma vez
    # O resultado será um array de scores como: [[0.0, 0.4, 0.0, 0.9, ...]]
    # Usamos [0] no final para pegar a lista de dentro
    todos_os_scores = cosine_similarity([query_vector], tfidf_matrix)[0]
    
    # 2. Junta cada índice com seu respectivo score
    scores = [(doc_id, score) for doc_id, score in enumerate(todos_os_scores)]
    
    # 3. Ordena os resultados do maior para o menor score
    return sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

def buscar_documentos(query_vector, tfidf_matrix, top_k):
    """Ranqueia os documentos mais relevantes para a query de forma vetorizada."""
    # 1. Passa a query como 2D e compara com a matriz inteira de uma vez
    todos_os_scores = cosine_similarity([query_vector], tfidf_matrix)[0]
    
    # 2. Pareamos cada índice com seu score, pegando apenas onde houve alguma similaridade (> 0)
    resultados = [
        (idx, score) for idx, score in enumerate(todos_os_scores) if score > 0
    ]
    
    # 3. Ordenamos do maior (mais relevante) para o menor
    resultados.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Retornamos apenas a quantidade pedida no top_k
    return resultados[:top_k]