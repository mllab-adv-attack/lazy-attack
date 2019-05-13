import numpy
import ortools.graph.pywrapgraph import SimpleMaxFlow


class GraphCutHelper(object):
  def __init__(self):
    pass

  def create_graph(self, edges):
    start_nodes = []
    end_nodes = []
    capacities = []

    self.graph = (start_nodes, end_nodes, capacities)

  def solve(self, source, sink):
    start_nodes, end_nodes, capacities = self.graph
    max_flow = SimpleMaxFlow()
    
    for start, end, capacity in zip(start_nodes, end_nodes, capacities):
      max_flow.AddArcWithCapacity(start, end, capacity)

