from collections import deque

# бинарный поиск
def binary_search(array, elem):
    """1) на выход отсортированный список, и значение
       2) если искомый элемент есть в списке, то возвращаем индекс элемента иначе null
       3) сложность log(2)_n """
    low = 0
    high = len(array)-1

    while low <= high:
        mid = (low + high) // 2
        guess = array[mid]
        if elem == guess:
            return mid
        elif elem < guess:
            high = mid - 1
        elif elem > guess:
            low = mid + 1
    return None


# поиск в ширину
def search_width(graph, item):
    search_queue = deque()
    start_node = list(graph.keys())[0]
    search_queue += graph[start_node]
    searched = []

    while search_queue:
        guess_item = search_queue.popleft()
        if guess_item not in searched:
            searched.append(guess_item)
            if guess_item == item:
                return True
            else:
                search_queue += graph[guess_item]
    return False






