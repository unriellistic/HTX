def test(a):
    if a == 3:
        print("finish")
        return
    a += 1
    print(a)
    test(a)

test(1)