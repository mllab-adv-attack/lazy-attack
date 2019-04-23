import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AffinityPropagation


'''
a = np.load('out/noise_nat_50000.npy')
a_flat = a.reshape(50000, -1)

ham = cdist(a_flat, a_flat, 'hamming')

np.save('out/ham_nat_50000.npy', ham)
print('noise saved!')
'''

ham = np.load('./out/ham_nat_50000.npy')

print('loading complete')
af = AffinityPropagation(affinity='precomputed').fit(ham)

print('clustering complete')


cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

np.save('./out/center_nat_50000.npy', cluster_centers_indices)
np.save('./out/label_nat_50000.npy', labels)
print('data saving complete')
