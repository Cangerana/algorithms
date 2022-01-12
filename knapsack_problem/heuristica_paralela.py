import numpy as np

from multiprocessing import Pool
from multiprocessing import shared_memory

from random import randint


def new_matrix(lenght):
    fliped = int(lenght * 0.1)
    new_m = np.full(fill_value=False, shape=lenght)

    for _ in range(fliped):
        pos = randint(0, lenght - 1)
        new_m[pos] = True
    return new_m


def greedy(items, max_weight):
    W = max_weight
    sel = []
    knapsack_weight = 0
    total_value = 0
    for i in range(len(items)):
        current_value = items[i][0]
        current_weight = items[i][1]
        if knapsack_weight + current_weight <= W:
            sel.append(items[i])
            knapsack_weight += current_weight
            total_value += current_value
        else:
            break

    return total_value


def flip(knackpack, index):
    return not knackpack[index] 


def best_impruvement(knackpack, items, weight_limit):
    max_generation = len(knackpack) * 0.05
    count_generation = 0

    best_global = status_knackpack(items, knackpack)

    while True:
        best_local = [0, 0]

        for index in range(knackpack.shape[0]):
            knackpack[index] = not knackpack[index]
            
            local_value, local_weight = status_knackpack(items, knackpack)

            if local_value > best_local[0] and local_weight <= weight_limit:
                best_local = [local_value, index]

            knackpack[index] = not knackpack[index]

        if best_local[0] > best_global[0]:
            best_global = best_local
            knackpack[best_local[1]] = not knackpack[best_local[1]]
            count_generation = 0
        else:
            count_generation += 1

        if count_generation > max_generation: break
    return knackpack


def status_knackpack(items, knackpack):
    value = sum(map(lambda i: i[0], items[knackpack]))
    weight = sum(map(lambda i: i[1], items[knackpack]))
    return value, weight


def dynamic(items, weight_limit):
    matrix = np.zeros(shape=(len(items) +1, weight_limit +1), dtype=np.int16)
    s_items = sorted(items, key=lambda x: x[1])

    for i in range(1, matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            if s_items[j-1][1] <= j:
                matrix[i][j] = max(
                    matrix[i-1][j],
                    s_items[j-1][0] + matrix[i-1][j-s_items[j-1][1]]
                )
            else:
                matrix[i][j] = matrix[i-1][j]

    return matrix[-1][-1]


def bp_search(v_w, weight_limit):
    m = np.zeros(shape=(len(v_w) +1, weight_limit + 1), dtype=np.int16)
    amt_lines = len(m)
    amt_cols = len(m[0])

    for i in range(1, amt_lines):
        wj = v_w[i - 1][1]
        vj = v_w[i - 1][0]

        for j in range(0, amt_cols):
            if wj > j:
                m[i][j] = m[i - 1][j]
            else:
                m[i][j] = max(m[i - 1][j],
                              vj + m[i - 1][j - wj])
    return m


def get_best_index(results, items, best_global):
    """Verifiaca qual dos resultados calculados em parlelo atingiu o melhor valor

    Return [int]: Index do melhor resultado ou -1 quando não supera o `best_global`
    """
    i = -1
    for j, r in enumerate(results):
        if best_global[0] < status_knackpack(items, r)[0]:
            i = j
    return i


def pipeline(size):
    """Faz o calculo da heuristica incial seguido do best improvement

    Parte assincrona do algoritmo.


    Params:
        size [Tuple]: tupla com o shape do vetor contido no share memory

    Return [np.array]: Array boleano que representa a melhor combinação,
    encontrada, de items na mochila
    """

    shm = shared_memory.SharedMemory(create=False, name='paa')
    params = np.ndarray(size, dtype=np.int64, buffer=shm.buf)

    knackpack = new_matrix(len(params[:-1]))

    mochila = best_impruvement(knackpack, params[:-1], params[-1][0])

    return mochila


def calcula():
    """Cria vetor de items, parmetros da mochila e os multiplos processos.
    Responsável por juntar os resultados dos multiplos processos e gerenciar
    o melhor resultado encontrado.

    Parte sincrona do algoritmo.
    """

    n_threads = 7

    items = [[randint(10, 50), randint(4, 10)] for _ in range(500)]
    items = np.array(items, dtype=np.int64)

    weight_limit = 400
    size = (len(items) + 1, 2)

    base_knackpack =  np.full(fill_value=False, shape=len(items))
    best_global = [0, 0]

    shm = shared_memory.SharedMemory(create=True, size=items.nbytes+512, name="paa")

    params = np.ndarray(size, dtype=items.dtype, buffer=shm.buf)
    params[:-1] = items
    params[-1][0] = weight_limit

    count =  0
    arr = [size] * n_threads
    
    with Pool(n_threads) as p:
        while True:
            results = p.map(pipeline, arr)

            index = get_best_index(results, items, best_global)
            
            if index == -1:
                if count >= 2:
                    break
                else:
                    count += 1
            else:
                base_knackpack = results[index]
                best_global = status_knackpack(items, base_knackpack)
                count = 0

            print(index, best_global)

    greedy_result = greedy(items, weight_limit)

    print("Resultado final Heuristica: ", best_global)
    print("Resultaddo guloso: ", greedy_result)
    print("Resultaddo dinamico: ", bp_search(items, weight_limit)[-1][-1])

    shm.close()
    shm.unlink()


if __name__ == '__main__':
    calcula()
