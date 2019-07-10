import numpy as np
import tensorflow as tf
import t3f

# Generate a random dense matrix of size 3 x 4.
a_dense = np.random.randn(3, 4)
# Convert the matrix into the TT-format with TT-rank = 3 (the larger the TT-rank,
# the more exactly the tensor will be converted, but the more memory and time
# everything will take). For matrices, matrix rank coinsides with TT-rank.
a_tt = t3f.to_tt_tensor(a_dense, max_tt_rank=3)
# a_tt stores the factorized representation of the matrix, namely it stores the matrix

# as a product of two smaller matrices which are called TT-cores. You can
# access the TT-cores directly.
print('factors of the matrix: ', a_tt.tt_cores)
# To check that the convertions into the TT-format didn't change the matrix too much,
# let's convert it back and compare to the original.
reconstructed_matrix = t3f.full(a_tt)
print('Original matrix: ')
print(a_dense)
print('Reconstructed matrix: ')
print(reconstructed_matrix)
