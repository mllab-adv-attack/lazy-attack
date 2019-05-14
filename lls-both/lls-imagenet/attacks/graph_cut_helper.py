import itertools
import numpy as np
from ortools.graph.pywrapgraph import SimpleMaxFlow


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
    
    return itertools.product(hs, ws)

  def create_graph(self, mask):
    self.height, self.width = np.shape(mask)
    num_nodes = self.height*self.width
    self.start_index = num_nodes
    self.end_index = num_nodes+1
    edges = {}
    
    xs, ys = np.where(mask==1)
    for x, y in zip(xs, ys): 
      index = x*self.width+y
      
      # unary
      edges[(self.start_index, index)] = 0
      edges[(index, self.end_index)] = 0
      
      # pairwise 
      neighbors = self._find_neighbors(x, y)      
      for neighbor in neighbors:   
        h, w = neighbor
        index_neighbor = h*self.width+w
        
        # this node need to be fixed
        if mask[h, w] == 0:
          edges[(self.start_index, index_neighbor)] = 0
          edges[(index_neighbor, self.end_index)] = 0
        
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
    
    if max_flow.Solve(self.start_index, self.end_index) == max_flow.OPTIMAL:
      source = max_flow.GetSourceSideMinCut()
      sink = max_flow.GetSinkSideMinCut()
      
      assignment = np.zeros([self.height, self.width], dtype=np.int32)
      
      for idx in source:
        if idx == self.start_index:
          continue
        h = idx // self.width
        w = idx % self.width
        assignment[h, w] = -1
      
      for idx in sink:
        if idx == self.end_index:
          continue
        h = idx // self.width
        w = idx % self.width
        assignment[h, w] = 1
      
      return assignment
    
    return None

