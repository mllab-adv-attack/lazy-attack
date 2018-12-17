from queue import PriorityQueue
import numpy as np

class Greedy:
  def __init__(self, loc, plus=0, minus=0, random=False):
    self.loc = loc
    self.plus = plus
    self.minus = minus
    self.random = random

  def __str__(self):
    return "location: {}, +perturb: {}, -perturb: {}".format(self.loc, self.plus, self.minus)

  def __eq__(self, other):
    if (isinstance(other, Greedy) and self.getVal() == other.getVal()):
      return True
    return False

  def __ne__(self, other):
    return not (self == other)

  #<, > changed for fitting to priority queue
  def __lt__(self, other):
    return self.getVal() < other.getVal()

  def __le__(self, other):
    return self.getVal() <= other.getVal()

  def __gt__(self, other):
    return self.getVal() > other.getVal()

  def __ge__(self, other):
    return self.getVal() >= other.getVal()

  def getLoc(self):
    return self.loc

  def getVal(self):
    if self.random is False:
      return min(self.plus, self.minus)
    else:
      return min(self.plus, self.minus)

  def getDir(self):
    if self.random is False:
      return self.plus <= self.minus
    else:
        #do not use!
      if self.plus == 0 and self.minus == 0:
        return 0.5
      elif self.plus == 0:
        return -np.sign(self.minus)
      elif self.minus == 0:
        return np.sign(self.plus)
      else:
        if np.sign(self.plus) != np.sign(self.minus):
          return np.sign(self.plus)
        else:
          if np.sign(self.plus) > 0:
            return self.plus / (self.plus + self.minus)
          else:
            return self.minus / (self.plus + self.minus)
          

  def update(self, plus, minus):
    self.plus = plus
    self.minus = minus

class ExpandedGreedy:
  def __init__(self, loc, mton=0, mtop=0, ntom=0, ntop=0, pton=0, ptom=0):
    self.loc = loc
    self.mton = mton
    self.mtop = mtop
    self.ntom = ntom
    self.ntop = ntop
    self.ptom = ptom
    self.pton = pton

  def __str__(self):
    return "location: {}, +perturb: {}, -perturb: {}".format(self.loc, self.plus, self.minus)

  def __eq__(self, other):
    if (isinstance(other, Greedy) and self.getVal() == other.getVal()):
      return True
    return False

  def __ne__(self, other):
    return not (self == other)

  #<, > changed for fitting to priority queue
  def __lt__(self, other):
    return self.getVal() < other.getVal()

  def __le__(self, other):
    return self.getVal() <= other.getVal()

  def __gt__(self, other):
    return self.getVal() > other.getVal()

  def __ge__(self, other):
    return self.getVal() >= other.getVal()

  def getLoc(self):
    return self.loc

  def getVal(self):
    return min(self.mton, self.mtop, self.ntom, self.ntop, self.ptom, self.pton)

  def getDir(self):
    return np.argmin([self.mton, self.mtop, self.ntom, self.ntop, self.ptom, self.pton])

  def update(self, mton, mtop, ntom, ntop, ptom, pton):
    self.mton = mton
    self.mtop = mtop
    self.ntom = ntom
    self.ntop = ntop
    self.ptom = ptom
    self.pton = pton

class PeekablePriorityQueue(PriorityQueue):
  def peek(self):
    try:
      with self.mutex:
        return self.queue[0]
    except IndexError:
      raise self.queue.Empty
