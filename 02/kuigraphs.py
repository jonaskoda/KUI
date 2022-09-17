# a simple KuiGraph class mimicking ai-gym interface
# for playing with graph/tree search algorithms
# Tomas Svoboda and the KUI team
# https://cw.fel.cvut.cz/wiki/courses/b3b33kui/start

GRAPH = dict()
GRAPH['S'] = [('d',3), ('c',1), ('a',1)]
GRAPH['a'] = [('e',1), ('b',2)]
GRAPH['b'] = [('f',1), ('e',1)]
GRAPH['c'] = [('d',1), ('f',1), ('b',1)]
GRAPH['d'] = [('G',1)]
GRAPH['e'] = [('f',1)]
GRAPH['f'] = []
GRAPH['G'] = []

class KuiGraph:
    def __init__(self, graph=None):
        if graph is None:
            self.graph = GRAPH
        self.path = None

    def reset(self):
        return 'S', 'G'

    def render(self,mode='human'):
        print(self.graph)
        if self.path is not None:
            print(self.path)

    def visualise(self):
        self.render()

    def set_path(self,path):
        self.path = path

    def expand(self, state):
        assert state in self.graph, "state is not known"
        return self.graph[state]



