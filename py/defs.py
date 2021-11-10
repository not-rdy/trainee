from random import choice


# Быстрая сортировка
def qsort(array):
    if len(array) < 2:
        return array
    else:
        pivot = choice(array)
        array.remove(pivot)
        less = [i for i in array if i <= pivot]
        greater = [i for i in array if i > pivot]
        return qsort(less) + [pivot] + qsort(greater)

# Рекурсивное суммирование
def sumList(arr):
    if len(arr) == 1:
        return arr[0]
    else:
        return arr[0] + sumList(arr[1:])

# Рекурсивный факториал
def fact(n):
    if n <= 1:
        return 1
    else:
        return n * fact(n-1)
