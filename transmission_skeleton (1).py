import numpy as np
from bitstring import BitArray

def readText(filename):
    with open(filename,'r') as file:
        data = file.read()
    return data

def convertText(string):
    c_list = list()
    b_list = list()

    for c in string:
        c_list.append(ord(c))
    for word in c_list:
        b_word = bin(word)[2:]
        for b in b_word:
            b_list.append(int(b))
    
    return b_list

def encodeHamming(message):
    encoded_m = list()
    quatuples = [message[4*i:4*i+4] for i in range(int(len(message)/4))]
    last = message[int(len(message)/4):len(message)]
    while len(last) < 4:
        len.append(0)

    # encode every quatuple using the (7,4,3)-Hamming Code and concatenate the encoded data to a message
    return [0,1]

def transmit(message, p):
    t_message = list()
    for b in message:
        if np.random.binomial(1,p):
            b = 0 if b == 1 else 1
        t_message.append(b)
    return t_message

def decodeHamming(message):
    decoded_m = list()
    septuples = [message[7*i:7*i+7] for i in range(int(len(message)/7))]

    #decode the septuples message using the (7,4,3)-Hamming code. 
    # Do error correction in case of errors.
    # concatenate the encoded data to a message
    return [1,0]

def countErrors(original, decoded):
    assert len(original) == len(decoded)
    equal = [original[i] == decoded[i] for i in range(0,len(original))]
    return equal.count(0)

if __name__ == "__main__":
    text = readText('text.txt')
    #Todo: implement the whole transmission chain using the implemented functions and print the resulting error rates
