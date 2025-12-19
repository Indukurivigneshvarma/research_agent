class ResearchTrace:
    def __init__(self):
        self.steps = []

    def add_iteration(self, iteration, query):
        self.steps.append(f"Iteration {iteration}: searched '{query}'")

    def mark_complete(self, reason):
        self.steps.append(f"Stopped: {reason}")

    def export(self):
        return "\n".join(self.steps)
