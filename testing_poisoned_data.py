import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import random

from CICDS_pipeline import cicidspipeline

# Step 1: Data Preparation with Poisoned Data
def generate_data():
    # Replace with actual data loading
    cipl = cicidspipeline()

    X_train, y_train, X_test, y_test = cipl.cicids_data_binary()

    
    # Introduce poisoned data
    num_poisoned = int(0.1 * len(X_train))  # 10% poisoned data
    poisoned_indices = np.random.choice(len(X_train), num_poisoned, replace=False)
    X_train[poisoned_indices] = np.random.rand(num_poisoned, 78)
    y_train[poisoned_indices] = 1 - y_train[poisoned_indices]  # Flip the labels

    return X_train, y_train, X_test, y_test, poisoned_indices

# Step 2: Antibody Initialization
def initialize_network(num_nodes, X_train, y_train):
    network = []
    for _ in range(num_nodes):
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_sample, y_sample = X_train[indices], y_train[indices]
        model = SVC(kernel='linear', probability=True)
        model.fit(X_sample, y_sample)
        network.append(model)
    return network

# Step 3: Affinity Calculation (minimizing false positive rate and identifying poisoned data)
def calculate_affinity(classifier, X_train, y_train, poisoned_indices):
    y_pred = classifier.predict(X_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    # Detect poisoned data
    poison_detection_accuracy = accuracy_score(y_train[poisoned_indices], y_pred[poisoned_indices])
    return 1 / (1 + fpr), poison_detection_accuracy  # Lower FPR and higher detection accuracy mean higher affinity

# Step 4: Network Dynamics
def update_network(network, X_train, y_train, num_clones, mutation_rate, poisoned_indices):
    affinities = [calculate_affinity(classifier, X_train, y_train, poisoned_indices)[0] for classifier in network]
    sorted_indices = np.argsort(affinities)[::-1]
    top_classifiers = [network[i] for i in sorted_indices[:num_clones]]
    
    clones = []
    edges = []
    for classifier in top_classifiers:
        for _ in range(num_clones):
            clone = SVC(kernel='linear', probability=True)
            noise = mutation_rate * np.random.randn(*X_train.shape)
            clone.fit(X_train + noise, y_train)
            clones.append(clone)
            edges.append((network.index(classifier), len(network) + len(clones) - 1))
    
    new_network = top_classifiers + clones
    return new_network[:len(network)], edges

# Step 5: Memory Update
def memory_update(network, X_train, y_train, memory_size, poisoned_indices):
    affinities = [calculate_affinity(classifier, X_train, y_train, poisoned_indices)[0] for classifier in network]
    sorted_indices = np.argsort(affinities)[::-1]
    memory = [network[i] for i in sorted_indices[:memory_size]]
    return memory

# Step 6: Visualize Network
def visualize_network(network, edges):
    G = nx.Graph()
    G.add_nodes_from(range(len(network)))
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
    plt.title('Artificial Immune Network of SVM Classifiers')
    plt.show()

# Main Function
def main():
    X_train, y_train, X_test, y_test, poisoned_indices = generate_data()
    num_nodes = 10
    num_clones = 5
    mutation_rate = 0.01
    memory_size = 10
    num_generations = 10

    network = initialize_network(num_nodes, X_train, y_train)
    all_edges = []

    for _ in range(num_generations):
        network, edges = update_network(network, X_train, y_train, num_clones, mutation_rate, poisoned_indices)
        all_edges.extend(edges)
        memory = memory_update(network, X_train, y_train, memory_size, poisoned_indices)
    
    best_classifier = memory[0]
    y_pred = best_classifier.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    test_accuracy = accuracy_score(y_test, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f'Test Accuracy: {test_accuracy:.2f}')
    print(f'False Positive Rate: {fpr:.2f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    
    # Evaluate poison detection accuracy
    poison_detection_accuracy = accuracy_score(y_train[poisoned_indices], best_classifier.predict(X_train[poisoned_indices]))
    print(f'Poison Detection Accuracy: {poison_detection_accuracy:.2f}')
    
    visualize_network(network, all_edges)

if __name__ == "__main__":
    main()
