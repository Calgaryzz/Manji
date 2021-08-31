from dataclasses import dataclass

@dataclass
class transition:
    label: str
    from_state: int
    to_state: str

    def __init__(self, label, to_state, from_state):
        self.label = label
        self.from_state = from_state
        self.to_state = to_state

    def to_string(self):
        print("label: " + self.label)
