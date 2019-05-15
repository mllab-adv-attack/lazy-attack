import itertools
import numpy as np
from ortools.graph.pywrapgraph import SimpleMaxFlow


class GraphCutHelper(object):
    def __init__(self, args):
        self.alpha = 10000000
        self.beta = 100
        self.radius = 1

    def _find_neighbors(self, x, y):
        height = self.height
        width = self.width
        radius = self.radius
        hs = np.arange(max(x-radius, 0), min(x+radius+1, height))
        ws = np.arange(max(y-radius, 0), min(y+radius+1, width))
        
        return itertools.product(hs, ws)

    def create_graph(self, mask, assignment, marginal_gain):
        self.height, self.width = np.shape(mask)
        num_nodes = self.height*self.width
        self.source_index = int(num_nodes)
        self.sink_index = int(num_nodes+1)
        edges = {}
    
        xs, ys = np.where(mask==1)
        for x, y in zip(xs, ys):
            x = int(x)
            y = int(y) 
            index = x*self.width+y
            unary = marginal_gain[x, y]
             
            source_capacity = unary if unary > 0 else 0
            sink_capacity = -unary if unary < 0 else 0

            # unary term
            edges[(self.source_index, index)] = int(self.alpha*source_capacity)
            edges[(index, self.sink_index)] = int(self.alpha*sink_capacity)
            
            # pairwise term
            neighbors = self._find_neighbors(x, y)      
            for neighbor in neighbors:   
                h, w = neighbor
                h = int(h)
                w = int(w)
                index_neighbor = h*self.width+w
              
                # this node need to be fixed
                if mask[h, w] == 0:
                    sign = assignment[h, w]
                    source_capacity = 1 if sign == -1 else 0
                    sink_capacity = 1 if sign == 1 else 0
                 
                    edges[(self.source_index, index_neighbor)] = int(self.alpha*source_capacity)
                    edges[(index_neighbor, self.sink_index)] = int(self.alpha*sink_capacity)
              
                edges[(index, index_neighbor)] = self.beta
                edges[(index_neighbor, index)] = self.beta

        self.graph = edges
      
    def solve(self):
        edges = self.graph
        max_flow = SimpleMaxFlow()
    
        for key, val in edges.items():
            start, end = key
            capacity = val
            max_flow.AddArcWithCapacity(start, end, capacity)
    
        if max_flow.Solve(self.source_index, self.sink_index) == max_flow.OPTIMAL:
            source = max_flow.GetSourceSideMinCut()
            sink = max_flow.GetSinkSideMinCut()
      
            assignment = np.zeros([self.height, self.width], dtype=np.int32)
      
            for idx in source:
                if idx == self.source_index:
                    continue
                h = idx // self.width
                w = idx % self.width
                assignment[h, w] = -1
      
            for idx in sink:
                if idx == self.sink_index:
                  continue
                h = idx // self.width
                w = idx % self.width
                assignment[h, w] = 1
      
            return assignment
    
        return None

