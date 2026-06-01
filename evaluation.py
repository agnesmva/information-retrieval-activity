from recuperar import buscar_documentos
from vector_model import calcular_tf_idf_query

def precision_at_k(ranking,
                   targets,
                   relevant_class,
                   k):

    top_k = ranking[:k]

    relevantes = sum(
        1
        for doc_id, _ in top_k
        if targets[doc_id] == relevant_class
    )

    return relevantes / k

def recall_at_k(ranking,
                targets,
                relevant_class,
                k=10):

    top_k = ranking[:k]

    recuperados = sum(
        1
        for doc_id, _ in top_k
        if targets[doc_id] == relevant_class
    )

    total_relevantes = sum(
        1
        for t in targets
        if t == relevant_class
    )

    return recuperados / total_relevantes

def average_precision(ranking, targets, relevant_class):
    """
    ranking: lista [(doc_id, score), ...]
    targets: classes reais dos documentos
    relevant_class: classe considerada relevante

    retorna AP da consulta
    """

    num_relevantes_encontrados = 0
    soma_precisoes = 0

    for posicao, (doc_id, _) in enumerate(ranking, start=1):

        if targets[doc_id] == relevant_class:

            num_relevantes_encontrados += 1

            precision = (
                num_relevantes_encontrados /
                posicao
            )

            soma_precisoes += precision

    if num_relevantes_encontrados == 0:
        return 0

    return soma_precisoes / num_relevantes_encontrados

def mean_average_precision(ap_scores):

    if len(ap_scores) == 0:
        return 0

    return sum(ap_scores) / len(ap_scores)

def avaliar_recuperacao_por_categoria(
    queries_textos, 
    queries_targets, 
    base_targets,
    funcao_preprocessamento,
    vocabulary, 
    term_to_idx, 
    idf_vector, 
    tfidf_matrix,
    categorias_alvo,
    nomes_categorias,
    top_k=20
):
    """
    Avalia um pipeline específico (ex: lemma) e retorna o AP médio para categorias específicas.
    """
    resultados_categoria = {}
    
    # Para cada categoria que queremos testar (ex: comp.graphics)
    for cat_nome in categorias_alvo:
        cat_id = nomes_categorias.index(cat_nome)
        
        # 1. Filtra as queries que pertencem a essa categoria específica
        indices_queries_cat = [i for i, target in enumerate(queries_targets) if target == cat_id]
        
        aps_desta_categoria = []
        
        # 2. Roda a busca para cada query dessa categoria
        for q_idx in indices_queries_cat:
            texto_query = queries_textos[q_idx]
            
            # Aplica o pré-processamento específico (original, stem ou lemma)
            query_tokens = funcao_preprocessamento(texto_query)
            
            # Vetoriza
            query_vector = calcular_tf_idf_query(query_tokens, vocabulary, term_to_idx, idf_vector)
            
            # Busca os top_k documentos
            resultados_busca = buscar_documentos(query_vector, tfidf_matrix, top_k=top_k)
            ids_recuperados = [idx for idx, score in resultados_busca]
            
            # 3. Monta o Gabarito (Ground Truth) para esta query
            # Relevante = 1 (mesma categoria da query), Irrelevante = 0
            # Como calcular_ap geralmente espera uma lista de 0s e 1s ou lista de relevantes:
            
            # Opção A: Se o seu calcular_ap espera uma lista de IDs relevantes:
            ids_relevantes_gabarito = [i for i, t in enumerate(base_targets) if t == cat_id]
            ranking_com_scores = [(idx, 1.0) for idx in ids_recuperados]
            ap_score = average_precision(ranking_com_scores, base_targets, cat_id)
            
            aps_desta_categoria.append(ap_score)
            
        # Calcula a média do AP (Mean Average Precision) para esta categoria
        map_categoria = sum(aps_desta_categoria) / len(aps_desta_categoria) if aps_desta_categoria else 0.0
        resultados_categoria[cat_nome] = round(map_categoria, 4)
        
    return resultados_categoria