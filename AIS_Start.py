from copy import deepcopy
import random
import numpy as np

def generate_random_features(num_features):
    # Generate random feature values within a specified range or distribution
    return [random.uniform(0, 1) for _ in range(num_features)]

class Antibody:
    def __init__(self, features):
        self.features = features
        self.affinity = 0  # Initialize affinity

class Antigen:
    def __init__(self, features):
        self.features = features

class ArtificialImmuneSystem:
    def __init__(self, antigen_data, num_antibodies, mutation_rate):
        self.antigens = [Antigen(features) for features in antigen_data]
        self.antibodies = [Antibody(generate_random_features(num_antibodies)) for _ in range(num_antibodies)]
        self.mutation_rate = mutation_rate


    def calculate_affinity(self, antibody, antigen):
        # Calculate affinity between antibody and antigen (e.g., Euclidean distance)
        # Calculate Euclidean distance between antibody and antigen features
        distance = np.linalg.norm(np.array(antibody.features) - np.array(antigen.features))
        # Normalize distance to be in range [0, 1]
        normalized_distance = 1 / (1 + distance)
        return normalized_distance

    def clonal_selection(self):
        for antigen in self.antigens:
            for antibody in self.antibodies:
                antibody.affinity = self.calculate_affinity(antibody, antigen)

            # Sort antibodies by affinity
            sorted_antibodies = sorted(self.antibodies, key=lambda x: x.affinity, reverse=True)

            # Select top antibodies for cloning based on some criteria (e.g., affinity)
            top_antibodies = sorted_antibodies[:num_clones]

            # Clone antibodies with mutation
            for antibody in top_antibodies:
                clone = deepcopy(antibody)
                self.mutate(clone)
                self.antibodies.append(clone)

    def mutate(self, antibody):
        for i in range(len(antibody.features)):
            # Apply mutation with a probability defined by mutation_rate
            if random.random() < self.mutation_rate:
                # Generate a random perturbation (e.g., Gaussian noise)
                perturbation = random.uniform(-1, 1)  # Example: random perturbation between -1 and 1
                # Apply the perturbation to the feature
                antibody.features[i] += perturbation
                # Ensure the feature remains within valid bounds if necessary
                # (Not implemented in this example)


    def detect_anomalies(self):
        anomalies = []
        for antigen in self.antigens:
            for antibody in self.antibodies:
                if self.calculate_affinity(antibody, antigen) >= threshold:
                    anomalies.append(antigen)
                    break
        return anomalies

# Example usage
antigen_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Example antigen data
num_antibodies = 100
mutation_rate = 0.1
threshold = 0.8
num_clones = 10

ais = ArtificialImmuneSystem(antigen_data, num_antibodies, mutation_rate)
ais.mutate(ais.antibodies[0])
ais.clonal_selection()
detected_anomalies = ais.detect_anomalies()
print("Detected anomalies:", detected_anomalies)
