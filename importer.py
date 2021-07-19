import graphviz

from models import node
import re


def construct_from_tfst(tfst, sentence_n):
    file = open(tfst, "r")
    stream = file.readlines()
    n_lines = int(stream[0])
    if sentence_n > n_lines:
        print("This sentence does not exist")
    else:
        d = dict
        pointer = 0
        length = len(stream)
        for i in range(length):
            if stream[i] == "$" + str(sentence_n) + '\n':
                print("Construction of the automaton for the sentence :", stream[i + 1])
                dot = graphviz.Digraph(comment=stream[i + 1])
                pointer = i
                break
        act_state = -4
        while stream[pointer] != "t\n":
            pointer = pointer + 1
            act_state = act_state + 1

        pointer = pointer - 1

        map = {}
        map[str(act_state)] = [node('t',None,act_state)]
        act_state = act_state - 1
        while re.search(r'^:', stream[pointer]) is not None:
            match = re.findall(r' ([0-9]*) ([0-9]*)', stream[pointer])
            for i in match:
                if str(act_state) in map.keys():
                    map[str(act_state)].append(node(i[0],map[i[1]],act_state))
                    dot.node(map[str(act_state)][len(map[str(act_state)])-1].label)
                    for n in map[str(act_state)][len(map[str(act_state)])-1].next_state:
                        dot.edge(map[str(act_state)][len(map[str(act_state)])-1].label, n.label)
                else:
                    map[str(act_state)] = [node(i[0],map[i[1]],act_state)]
                    dot.node(map[str(act_state)][0].label)
                    for n in map[str(act_state)][0].next_state:
                        dot.edge(map[str(act_state)][0].label, n.label)
            act_state = act_state -1
            pointer = pointer - 1
        first_node = node('0',map['0'],0)
        for n in first_node.next_state:
            dot.node(n.label)
            dot.edge(first_node.label, n.label)
        dot.render('output/'+str(sentence_n), view=True)
        return first_node