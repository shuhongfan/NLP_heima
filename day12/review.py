def fun(a, b1, c):
    print(f'a-->{a}')
    print(f'b-->{b1}')
    print(f'c-->{c}')

# fun(a=2, b=3, c=4)
dict1 = {"a":2, "b1":3, "c":4}
fun(**dict1)