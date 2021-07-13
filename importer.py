from models import node
from models import automaton
import re

def construct_from_tfst(tfst,line):
    file = open(tfst,"r")
    stream = file.readlines()
    n_lines = int(stream[0])
    if line > n_lines :
        print("This sentence does not exist")
    else:
        d = dict
        pointer = 0
        length = len(stream)
        for i in range(length):
            if stream[i] == "$" + str(line) + '\n':
                print("Construction of the automaton for the sentence :", stream[i+1])
                a = automaton(stream[i+1], None)
                pointer = i
                break

        while stream[pointer] != "t\n":
            pointer = pointer+1

        pointer = pointer - 1
        while re.search(r'^:', stream[pointer]) is not None:
            for j in stream[pointer]:
                print(j) #Travail à faire ici
            pointer = pointer - 1


