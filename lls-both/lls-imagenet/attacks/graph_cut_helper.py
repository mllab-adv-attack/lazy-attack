import itertools
import numpy as np
import ortools.graph.pywrapgraph import SimpleMaxFlow


class GraphCutHelper(object):
  def __init__(self, args):
    self.alpha = args.alpha
    self.beta = args.beta

  def _find_neighbors(self, x, y):
    height = self.height
    width = self.width
    radius = self.radius
    hs = np.arange(max(x-radius, 0), min(x+radius, height))
    ws = np.arange(max(y-radius, 0), min(y+radius, width))
    
    return itertools.product(hs, ys): 

  def create_graph(self, mask):
    self.height, self.width = np.shape(mask)
    num_nodes = self.height*self.width
    start_index = num_nodes
    end_index = num_nodes+1
    edges = {}
    
    xs, ys = np.where(mask==1)
    for x, y in zip(xs, ys): 
      index = x*self.width+y
      
      # unary
      edges[(start_index, index)] = 0
      edges[(index, end_index)] = 0
      
      # pairwise 
      neighbors = self._find_neighbors(x, y)      
      for neighbor in neighbors:   
        h, w = neighbor
        index_neighbor = h*self.width+w
        
        # this node need to be fixed
        if mask[h, w] == 0:
          edges[(start_index, index_neighbor)] = 0
          edges[(index_neighbor, end_index)] = 0
        
        edges[(index, index_neighbor)] = self.beta
        edges[(index_neighbor, index)] = self.beta

    self.graph = edges
    
  def solve(self, source, sink):
    edges = self.graph
    max_flow = SimpleMaxFlow()
    
    for key, val in edges.items():
      start, end = key
      capacity = val
      max_flow.AddArcWithCapacity(start, end, capacity)

