a = [1, 2, 3]
b = [6, 7, 8, 9]
c = [12, 4]
d = [a, b, c]

idx = 0
flag = False
for values in d:
    idx += 1
    print(values)
    for value in values:
        if 6 == value:
            flag = True
            print(value)
            break
    if flag:
        break
print(idx)
