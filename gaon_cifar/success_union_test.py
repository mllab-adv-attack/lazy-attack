from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce
import numpy as np

ldg = np.reshape(np.load('./out/ldg_success.npy'), -1)
ldg_dec = np.reshape(np.load('./out/ldg_dec_success.npy'), -1)
pgd = np.reshape(np.load('./out/pgd_success.npy'), -1)

print('ldg accuracy: {:.2f}%'.format((500-len(ldg))/500*100))
print('ldg_dec accuracy: {:.2f}%'.format((500-len(ldg_dec))/500*100))
print('pgd accuracy: {:.2f}%'.format((500-len(pgd))/500*100))

print()
print('ldg successes:', len(ldg))
print('ldg_dec successes:', len(ldg_dec))
print('pgd successes:', len(pgd))


print()
print('ldg&ldg_dec successes:', len(np.intersect1d(ldg, ldg_dec)))
print('ldg&pgd successes:', len(np.intersect1d(ldg, pgd)))
print('pgd&ldg_dec successes:', len(np.intersect1d(pgd, ldg_dec)))

print()
print('ldg&ldg_dec&pgd successes:', len(reduce(np.intersect1d, (ldg, ldg_dec, pgd))))

print()
print('ldg|ldg_dec successes:', len(np.union1d(ldg, ldg_dec)))
print('ldg|pgd successes:', len(np.union1d(ldg, pgd)))
print('pgd|ldg_dec successes:', len(np.union1d(pgd, ldg_dec)))

print()
print('ldg|ldg_dec|pgd successes:', len(reduce(np.union1d, (ldg, ldg_dec, pgd))))

print()
print('ldg, ldg_dec similarity: {:.2f}%'.format(len(np.intersect1d(ldg, ldg_dec))/len(np.union1d(ldg,ldg_dec))*100))
print('ldg, pgd similarity: {:.2f}%'.format(len(np.intersect1d(ldg, pgd))/len(np.union1d(ldg,pgd))*100))
print('pgd, ldg_dec similarity: {:.2f}%'.format(len(np.intersect1d(pgd, ldg_dec))/len(np.union1d(pgd,ldg_dec))*100))
print('lgd, pgd, ldg_dec similarity: {:.2f}%'.format(len(reduce(np.intersect1d, (ldg, pgd, ldg_dec)))/len(reduce(np.union1d,(ldg,pgd,ldg_dec)))*100))

