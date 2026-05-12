from collections import deque

n, k = map(int, input().split())
arrays = [list(map(int, input().split())) for _ in range(n)]
int_list = list(map(int, input().split()))

def get_array(k):
    for array, val in zip(arrays, int_list):
        length = len(array) - 1
        for _ in range(val):
            if k > length:
                k -= length
            else:
                return array[k]

print(get_array(k))