import time
import numpy as np

fibo1 = 0
fibo2 = 1

start = time.time()

n = 0

while time.time() - start < 1:
    keep = fibo2

    fibo2 = fibo1 + fibo2

    fibo1 = keep

    n += 1

print(n, fibo2, fibo1)