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

  def create_graph(self, mask, noise, latest_gain):
    self.height, self.width = np.shape(mask)
    num_nodes = self.height*self.width
    self.source_index = num_nodes
    self.sink_index = num_nodes+1
    edges = {}
    
    xs, ys = np.where(mask==1)
    for x, y in zip(xs, ys): 
      index = x*self.width+y
      gain = latest_gain[x, y]
      
      """Should be fixed, check if gain is positive or negative"""
      source_capacity = 0
      sink_capacity = 0

      # unary term
      edges[(self.source_index, index)] = int(self.alpha*source_capacity)
      edges[(index, self.sink_index)] = int(self.alpha*sink_capacity)
      
      # pairwise term
      neighbors = self._find_neighbors(x, y)      
      for neighbor in neighbors:   
        h, w = neighbor
        index_neighbor = h*self.width+w
        
        # this node need to be fixed
        if mask[h, w] == 0:
          """Should be fixed, check the sign of noise""" 
          source_capacity = 0 
          sink_capacity = 0
           
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

