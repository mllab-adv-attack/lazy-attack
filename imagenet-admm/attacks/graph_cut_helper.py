import math
import numpy as np
from ortools.graph.pywrapgraph import SimpleMaxFlow
from scipy.spatial import KDTree 
import tensorflow as tf

class GraphCutHelper(object):
  def __init__(self, args):
    self.alpha = args.alpha
    self.beta = args.beta
  
  def create_graphs(self, nodes):  
    self.graphs = [None]*3

    for c in range(3):
      nodes_channel = nodes[c]
      if len(nodes_channel) == 0:
        continue
      num_nodes = len(nodes_channel)+2
      positions = np.zeros([0, 2], np.int32) 
      pos_to_idx = {}
      idx_to_pos = {}
      start_nodes = []
      end_nodes = []
      capacities = []

      # Construct KD-Tree and calculate unary term
      for node in nodes_channel:
        idx, x, y, unary_source, unary_sink = node
        positions = np.concatenate([positions, np.reshape([x, y], [1, -1])], axis=0) 
        pos_to_idx[(x, y)] = idx
        idx_to_pos[idx] = (x, y)
        # From source
        start_nodes.append(0)
        end_nodes.append(idx)
        capacities.append(int(self.alpha*unary_source))
        # To sink
        start_nodes.append(idx)
        end_nodes.append(num_nodes-1)
        capacities.append(int(self.alpha*unary_sink))
   
      kdtree = KDTree(positions)
    
      # Calculate pairwise term 
      for node in nodes_channel:
        idx, x, y, _, _ = node
        neighbors = kdtree.query_ball_point([x, y], r=2, p=math.inf)
        for neighbor in neighbors:
          x_n, y_n = kdtree.data[neighbor]
          idx_n = pos_to_idx[(x_n, y_n)]
          if idx_n <= idx:
            continue
          # Intermediate edges
          start_nodes.append(idx)
          end_nodes.append(idx_n)
          capacities.append(self.beta)
          
          start_nodes.append(idx_n)
          end_nodes.append(idx)
          capacities.append(self.beta)
      
      graph = [start_nodes, end_nodes, capacities, num_nodes, idx_to_pos]
      self.graphs[c] = graph
    
  def solve(self):
    results = [[] for _ in range(3)]
    for c in range(3):
      graph = self.graphs[c]
      if graph is None:
        continue
      start_nodes, end_nodes, capacities, num_nodes, idx_to_pos = graph
      max_flow = SimpleMaxFlow()
      for i in range(len(start_nodes)):
        max_flow.AddArcWithCapacity(start_nodes[i], end_nodes[i], capacities[i])
      if max_flow.Solve(0, num_nodes-1) == max_flow.OPTIMAL:
        source = max_flow.GetSourceSideMinCut()
        sink = max_flow.GetSinkSideMinCut()
        for idx in source[1:]:    
          x, y = idx_to_pos[idx]
          results[c].append([idx, x, y, -1])
        for idx in sink[1:]:
          x, y = idx_to_pos[idx] 
          results[c].append([idx, x, y, 1]) 
    
    return results
     
     
     
      
