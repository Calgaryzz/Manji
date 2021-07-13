class node:
    def __init__(self, label, next_state, state):
        node.label = label
        node.next_state = next_state
        node.state = state

class automaton:
    def __init__(self, sentence, values):
        automaton.sentence = sentence
        automaton.values = values