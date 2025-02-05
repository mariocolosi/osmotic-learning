# v0.1.0

from simulator.misc.utils import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools


def plot_training_loss(losses, figsize=(8, 6), fontsize=10):
    """
    Plot training loss across epochs.
    """
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    plt.plot(losses, marker='o', linestyle='-', label='Training Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=fontsize, labelpad=20)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.legend()
    plt.grid(True)
    plt.title('Training Loss Over Epochs')
    plt.show()


def plot_accuracy(metrics, figsize=(8, 6), fontsize=10, ylim=(0.8, 1.01)):
    """
    Plot accuracy across epochs.
    """
    accuracy = metrics["accuracy"]
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    plt.plot(accuracy, marker='o', linestyle='-', markersize=10, label='Ctx accuracy', linewidth=2)
    markers_assignment = {}
    markers = itertools.cycle(['s', 'o','x', 'D', 'v', '<', '>', 'p', '*', 'h'])
    colors = itertools.cycle(['orange', 'green', 'red', 'purple', 'gray', 'brown', 'pink', 'olive', 'cyan', 'blue', ])

    for id, value in metrics["clusters_accuracy"].items():
        epochs = value['epoch']
        accuracies = value['accuracy']

        segment_epochs = []
        segment_accuracies = []

        for i in range(len(epochs)):
            if id not in markers_assignment:
                marker = next(markers) if id != 'unclustered' else '^'
                markers_assignment[id] = {
                    'marker': marker,
                    'color': next(colors)
                }
            else:
                marker = markers_assignment[id]['marker']
                color = markers_assignment[id]['color']

            if i == 0 or epochs[i] == epochs[i - 1] + 1:
                segment_epochs.append(epochs[i])
                segment_accuracies.append(accuracies[i])
            else:
                if segment_epochs:
                    label = f'Sub-ctx {id}' if id != 'unclustered' else 'Unclustered'
                    plt.plot(segment_epochs, segment_accuracies, marker=marker, markersize=10, color=color, linestyle='--', label=label, linewidth=1)
                segment_epochs = [epochs[i]]
                segment_accuracies = [accuracies[i]]

        if segment_epochs:
            if id not in markers_assignment:
                marker = next(markers) if id != 'unclustered' else '^'
                markers_assignment[id] = {
                    'marker': marker,
                    'color': next(colors)
                }
            else:
                marker = markers_assignment[id]['marker']
                color = markers_assignment[id]['color']
            label = f'Sub-ctx {id}' if id != 'unclustered' else 'Unclustered'
            plt.plot(segment_epochs, segment_accuracies, marker=marker, color=color, markersize=10, linestyle='--', label=label, linewidth=1)

    plt.xlabel('Epoch', fontsize=fontsize, labelpad=20)
    plt.ylabel('Accuracy', fontsize=fontsize, labelpad=20)
    plt.ylim(ylim[0], ylim[1])
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())
    plt.grid(True)
    plt.title('Accuracy Over Epochs')
    plt.show()


def plot_embeddings_similarity_matrix(embeddings, figsize=(6,4), fontsize=10, tickevery=30, min=0, max=1, cbar=False):
    keys = sorted(list(embeddings.keys()))
    n_clients = len(keys)
    
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            client_a, client_b = keys[i], keys[j]
            emb_a, emb_b = embeddings[client_a], embeddings[client_b]
            
            embedding_labels_x = [f"{idx}" for idx in range(len(emb_a))]
            embedding_labels_y = [f"{idx}" for idx in range(len(emb_b))]
            
            similarity_matrix = pd.DataFrame(index=embedding_labels_y, columns=embedding_labels_x, dtype=float)
            
            for idx_a, emb_vec_a in enumerate(emb_a):
                for idx_b, emb_vec_b in enumerate(emb_b):
                    similarity = cosine_similarity(emb_vec_a, emb_vec_b)
                    label_a = embedding_labels_x[idx_a]
                    label_b = embedding_labels_y[idx_b]
                    similarity_matrix.at[label_b, label_a] = similarity
            
            similarity_matrix = similarity_matrix.astype(float)
            
            plt.figure(figsize=figsize)
            plt.rcParams.update({'font.size': fontsize})
            sns.heatmap(similarity_matrix, annot=False, cmap='viridis', fmt=".2f", cbar=cbar, vmin=min, vmax=max, )
            plt.xlabel(f"Agent-{client_a} (t)", fontsize=fontsize, labelpad=20)
            plt.ylabel(f"Agent-{client_b} (t)", fontsize=fontsize)
            ax = plt.gca()
            ax.set_xticks(np.arange(0, len(embedding_labels_x), tickevery))
            ax.set_yticks(np.arange(0, len(embedding_labels_y), tickevery))
            ax.set_xticklabels(embedding_labels_x[::tickevery], rotation=0)
            ax.set_yticklabels(embedding_labels_y[::tickevery])

            num_elements = len(embedding_labels_x)
            ax.plot(range(num_elements), range(num_elements), linestyle='--', dashes=(8, 8), color='red', linewidth=2)
            plt.title(f"Embeddings Similarity Matrix")
            plt.show()


def plot_similarity_matrix(similarity_matrix, figsize=(6,4), fontsize=10, min=0, max=1):
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', fmt=".2f", cbar=False, vmin=min, vmax=max)
    plt.xlabel('Agent', fontsize=fontsize, labelpad=20)
    plt.ylabel('Agent', fontsize=fontsize)
    plt.title('Embeddings Similarity Matrix')
    plt.show()


def print_cluster_info(clusters):
    """
    Print information about the clusters.
    """
    printed_clusters_agent_id = {}
    for cluster_id, cluster in clusters.items():
        for agent in cluster:
            try: printed_clusters_agent_id[cluster_id].append(agent.agent_id)
            except KeyError: printed_clusters_agent_id[cluster_id] = [agent.agent_id]
    print(f"[INFO] Clusters: {printed_clusters_agent_id}")