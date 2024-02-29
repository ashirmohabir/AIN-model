import numpy as np

def calculate_affinity(antibody_features, antigen_features):
    # Convert features to NumPy arrays for convenient computation
    antibody_array = np.array(antibody_features)
    antigen_array = np.array(antigen_features)
    
    # Calculate Euclidean distance between antibody and antigen features
    distance = np.linalg.norm(antibody_array - antigen_array)
    
    # Normalize distance to be in range [0, 1]
    normalized_distance = 1 / (1 + distance)
    
    return normalized_distance

# Example usage:
antibody_features = [1, 2, 3]
antigen_features = [4, 5, 6]
affinity = calculate_affinity(antibody_features, antigen_features)
print("Affinity:", affinity)
