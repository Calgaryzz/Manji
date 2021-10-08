import graphviz
import networkx as nx
import matplotlib.pyplot as plt

from data_creation.automaton import transition
import re


"""import a single lattice from text.tsft"""
def construct_from_tfst(tfst, sentence_n, visual=False):
    lattice = nx.DiGraph()
    file = open(tfst, "r")
    stream = file.readlines()
    n_sentences = int(stream[0])
    if sentence_n > n_sentences:
        print("This sentence does not exist")
    else:
        d = dict
        pointer = 0
        length = len(stream)
        for i in range(length):
            if stream[i] == "$" + str(sentence_n) + '\n':
                if visual:
                    print("Construction of the automaton for the sentence :", stream[i + 1])
                    dot = graphviz.Digraph(comment=stream[i + 1])
                else:
                    print(sentence_n,"/", n_sentences)
                pointer = i
                break
        act_state = -4
        while stream[pointer] != "t\n":
            pointer = pointer + 1
            act_state = act_state + 1

        pointer = pointer - 1

        map = {}
        map[str(act_state)] = [transition('t',None,act_state)]
        act_state = act_state - 1
        for i in range(act_state+1):
            map[str(i)] = []
        while re.search(r'^:', stream[pointer]) is not None:
            match = re.findall(r' ([0-9]*) ([0-9]*)', stream[pointer])
            matchstd = re.findall(r'@\{*} ([0-9]*)', stream[pointer])
            for i in match:
                print(i)
                map[str(act_state)].append(transition(i[0],map[i[1]],act_state))
                lattice.add_node(map[str(act_state)][len(map[str(act_state)]) - 1].label)
                if visual:
                    dot.node(map[str(act_state)][len(map[str(act_state)])-1].label)
                for n in map[str(act_state)][len(map[str(act_state)])-1].to_state:
                    if visual:
                        dot.edge(map[str(act_state)][len(map[str(act_state)])-1].label, n.label)
                    lattice.add_edge(map[str(act_state)][len(map[str(act_state)])-1].label, n.label)
            act_state = act_state -1
            pointer = pointer - 1
        first_transition = transition('0',map['0'],0)

        for n in first_transition.to_state:
            if visual:
                dot.node(n.label)
                dot.edge(first_transition.label, n.label)
                dot.render('output/' + str(sentence_n), view=True)
            lattice.add_node(n.label)
            lattice.add_edge(first_transition.label, n.label)
        return lattice


def construct_corpus(tfst):
    """
    Create a corpus with all the sentences of the .tfst and chose the path randomly
    todo: don't use construct_from_tsft, because it is reopening and reclosing stream everytime
    """

    file = open(tfst, "r")
    stream = file.readlines()
    n_lines = int(stream[0])
    for i in range(1, n_lines+1):
        construct_from_tfst(tfst=tfst,sentence_n=i)
