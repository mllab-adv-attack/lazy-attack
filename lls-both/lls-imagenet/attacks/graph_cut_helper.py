import itertools
import numpy as np
from ortools.graph.pywrapgraph import SimpleMaxFlow

LARGE_INT = int(1e+15)

class GraphCutHelper(object):
    def __init__(self, args):
        self.alpha = args.alpha
        self.beta = args.beta
        self.radius = args.radius

        self.block_size = args.lls_block_size

        self.height, self.width = None, None
        self.start_index, self.end_index = None, None
        self.graph = None

    @staticmethod
    def _array_downsize(arr, block_size):
        height, width = arr.shape

        return np.copy(arr[::block_size, ::block_size])

    @staticmethod
    def _array_upsize(arr, block_size):
        height, width = arr.shape

        return arr.repeat(block_size, axis=0).repeat(block_size, axis=1)

    def _find_neighbors(self, x, y):
        height = self.height
        width = self.width
        radius = self.radius
        hs = np.arange(max(x-radius, 0), min(x+radius+1, height))
        ws = np.arange(max(y-radius, 0), min(y+radius+1, width))

        return itertools.product(hs, ws)

    def set_block_size(self, block_size):
        self.block_size = block_size

    def create_graph(self, mask, latest_gain, noise):
        # downscaling
        mask_rs = self._array_downsize(mask, self.block_size)
        latest_gain_rs = self._array_downsize(latest_gain, self.block_size)
        noise_rs = self._array_downsize(noise, self.block_size)

        self.height, self.width = np.shape(mask_rs)
        num_nodes = self.height*self.width
        self.start_index = num_nodes
        self.end_index = num_nodes+1
        edges = {}

        xs, ys = np.where(mask_rs == 1)
        for x, y in zip(xs, ys):
            index = x*self.width+y

            # unary
            if latest_gain_rs[x, y] >= 0:
                edges[(self.start_index, index)] = int(self.alpha * latest_gain_rs[x, y])
                edges[(index, self.end_index)] = 0
            else:
                edges[(self.start_index, index)] = 0
                edges[(index, self.end_index)] = -int(self.alpha * latest_gain_rs[x, y])

            # pairwise
            neighbors = self._find_neighbors(x, y)
            for neighbor in neighbors:
                h, w = neighbor
                index_neighbor = h*self.width+w

                # this node need to be fixed
                if mask_rs[h, w] == 0:
                    if noise_rs[h, w] > 0:
                        edges[(self.start_index, index_neighbor)] = 0
                        edges[(index_neighbor, self.end_index)] = LARGE_INT
                    else:
                        edges[(self.start_index, index_neighbor)] = LARGE_INT
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
            
            max_flow.AddArcWithCapacity(int(start), int(end), capacity)

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

            # upscaling
            assignment = self._array_upsize(assignment, self.block_size)

            return assignment

        print('failed...')
        return None

