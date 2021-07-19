from dataclasses import dataclass


@dataclass
class node:
    label: str
    state: int
    next_state: str

    def __init__(self, label, next_state, state):
        self.label = label
        self.state = state
        self.next_state = next_state

    def to_string(self):
        print("label: " + self.label + " ,state: ", self.state, " ,next state :" + self.next_state)
