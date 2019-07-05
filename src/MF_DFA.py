class MF_DFA:

    def __init__(self, X, num_points=10):
        self.X = X
        self.Y = None
        self.Fn = None
        self.N = len(self.X)
        self.size_boxs = None
        self.num_points = num_points

    def run(self):
        pass