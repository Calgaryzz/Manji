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
        act_state = -3
        while stream[pointer] != "t\n":
            pointer = pointer+1
            act_state = act_state + 1

        pointer = pointer - 1
        act_state = act_state -1

        map = {}
        while re.search(r'^:', stream[pointer]) is not None:
            match = re.findall(r' ([0-9]*) ([0-9]*)', stream[pointer])
            for j in range(len(match)):
                if act_state not in map.keys():
                    map[act_state] = [node(match[j][0],match[j][1],act_state)]
                else:
                    map[act_state].append(node(match[j][0],match[j][1],act_state))
            pointer = pointer - 1
            act_state = act_state - 1
            for n in map.items():
                print(n[1][0].next_state)
