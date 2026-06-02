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
    top_k=20
):
    resultados_categoria = {}

    for categoria in categorias_alvo:

        print(f"Processando categoria: {categoria}")

        # seleciona apenas as queries desta categoria
        indices_queries = [
            i
            for i, target in enumerate(queries_targets)
            if target == categoria
        ]

        aps_categoria = []

        for idx_query in indices_queries:

            texto_query = queries_textos[idx_query]

            query_tokens = funcao_preprocessamento(texto_query)

            query_vector = calcular_tf_idf_query(
                query_tokens,
                vocabulary,
                term_to_idx,
                idf_vector
            )

            resultados_busca = buscar_documentos(
                query_vector,
                tfidf_matrix,
                top_k=top_k
            )

            ap = average_precision(
                resultados_busca,
                base_targets,
                categoria
            )

            aps_categoria.append(ap)

        resultados_categoria[categoria] = (
            round(sum(aps_categoria) / len(aps_categoria), 4)
            if aps_categoria
            else 0.0
        )

    return resultados_categoria