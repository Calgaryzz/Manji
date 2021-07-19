from models import node
import re


def construct_from_tfst(tfst, sentence):
    file = open(tfst, "r")
    stream = file.readlines()
    n_lines = int(stream[0])
    if sentence > n_lines:
        print("This sentence does not exist")
    else:
        d = dict
        pointer = 0
        length = len(stream)
        for i in range(length):
            if stream[i] == "$" + str(sentence) + '\n':
                print("Construction of the automaton for the sentence :", stream[i + 1])
                #a = automaton(stream[i + 1], None)
                pointer = i
                break
        act_state = -4
        while stream[pointer] != "t\n":
            pointer = pointer + 1
            act_state = act_state + 1

        pointer = pointer - 1
        act_state = act_state - 1

        map = {}
        while re.search(r'^:', stream[pointer]) is not None:
            match = re.findall(r' ([0-9]*) ([0-9]*)', stream[pointer])
            for i in match:
                if act_state in map.keys():
                    map[act_state].append(node(i[0],i[1],act_state))
                else:
                    map[act_state] = [node(i[0],i[1],act_state)]
            act_state = act_state -1
            pointer = pointer - 1
        for k in map.keys():
            for t in map[k]:
                print(t.to_string())