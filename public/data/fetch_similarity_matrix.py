import numpy as np
import json

# Load and convert
matrix = np.load('similarity_matrix.npy')
matrix_list = matrix.tolist()  # convert numpy array to list

# Save as JSON
with open('similarity_matrix.json', 'w') as f:
    json.dump(matrix_list, f)
