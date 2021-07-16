class node:
    label = None
    next_state = None
    state = None

    def __init__(self, label, next_state, state):
        node.label = label
        node.next_state = next_state
        node.state = state

    def to_string(self):
        return 'label: ' + node.label + ', next state: ' + node.next_state

class automaton:
    def __init__(self, sentence, values):
        automaton.sentence = sentence
        automaton.values = values