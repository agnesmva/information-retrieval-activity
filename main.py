from pre_processamento import *
from data_downloading import *

if __name__ == "__main__":
    # Baixar os dados
    data_set_completo = download_data_completa()
    colecao_subset_removida = download_data_subset_removida()

    print("Texto orginal sendo lematizado...")
    lema_subset = [preprocess_lemma(doc) for doc in colecao_subset_removida.data]
    print("Texto orginal sendo processado...")
    original_subset = [preprocess_original(doc) for doc in colecao_subset_removida.data]
    print("Texto sendo steaminzado...")
    steaming_subset = [preprocess_stemming(doc) for doc in colecao_subset_removida.data]
    print("Finalizado o pré-processamento.")

    print("Steamming:")
    print(sum(len(doc) for doc in steaming_subset))
    print("Original:")
    print(sum(len(doc) for doc in original_subset))
    print("Lema:")
    print(sum(len(doc) for doc in lema_subset))