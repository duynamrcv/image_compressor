import numpy as np

def binary_encode(codes):
    code_file = open('binary.txt', "w")
    count = 0

    for code in codes:
        x = str(code).split('.')[1][:13][::-1]
        b = bin(int(x))
        out = b[2:]
        # print(b[2:])
        # print(len(b[2:]))
        code_file.write(out + '\n')
        count += len(out)
    print("Encode size: {}kb".format(count/8/1024))
    return count

def binary_decode(code_path='binary.txt'):
    code_file = open(code_path, 'r')
    for code in code_file:
        print(type(code))
        print(type(str(code)))
        # code ='0b' + code[:-1]
        # x = 
        # print(x)

binary_decode()

# codes = np.load("encode.npy")

# codes = [0.05, 0.757, 0.98937]

