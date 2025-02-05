# v0.1.0

import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from simulator.misc.utils import set_seed, cosine_similarity
from simulator.misc.visualization import plot_similarity_matrix, print_cluster_info


class Diffuser:
    def __init__(self, agents, epochs=5, clustering=True, clustering_step = 3, device='cpu'):
        """
        Initialize the diffuser with agents, epochs, and device.
        """
        self.agents = agents
        self.clusters = {
            0: agents.values()
        }
        self.similarity_matrix = None
        self.epochs = epochs
        self.device = device
        self.clustering = clustering
        self.clustering_step = clustering_step
        self.global_embedding = None
        self.metrics = {"training_losses": [], "accuracy": [], "clusters_accuracy": {}}
        self.local_embeddings = {}
        set_seed(42)


    def initialize_global_embedding(self):
        """
        Initialize the global embedding with the same shape as the agent embeddings.
        """
        shapes = [agent.get_embedding_shape() for agent in self.agents.values()]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All agent models must have the same output embedding shape.")

        self.global_embedding = torch.zeros(shapes[0]).to(self.device)
        print(f"[DIFFUSER] Global embedding initialized with shape: {shapes[0]}")


    def start_agents(self):
        """
        Start the agent processes.
        """
        self.processes = [
            mp.Process(target=agent.run) for agent in self.agents.values()
        ]

        for process in self.processes:
            process.start()
        print("[DIFFUSER] All agent processes started.")


    def stop_agents(self):
        """
        Stop the agent processes.
        """
        for _, cluster in self.clusters.items():
            for agent in cluster:
                agent.receiving_queue.put({'action': 'STOP'}) 

        for process in self.processes:
            process.join()
        print("[DIFFUSER] All agent processes stopped.")


    def wait_agents_messages(self, type, cluster = None):
        """
        Wait for all agents in the cluster to send a message of the specified type.
        """
        if not cluster:
            cluster = [agent for agent in self.agents.values()]

        results = {}
        completed_agents = 0
        while completed_agents < len(cluster):
            for agent in cluster:
                message = agent.sending_queue.get()
                if message['action'] == type:
                    results[message['metadata']['agent_id']] = message['data']
                    completed_agents += 1
                else:
                    raise ValueError(f"[DIFFUSER] Invalid message received from agent: expected {type}, got {message['action']}.")
        return results


    def compute_accuracy(self, epoch):
        """
        Compute the accuracy of the agent embeddings.
        """
        for agent in self.agents.values():
            agent.receiving_queue.put({'action': 'EVALUATE'})

        avg_accuracies = []
        for _, cluster in self.clusters.items():
            local_embeddings = self.wait_agents_messages('EVALUATE_COMPLETED', cluster)
            _, full_similarity_matrix = self.compute_similarity_matrix(local_embeddings, len(list(local_embeddings.values())[0]))
            avg_similarity = full_similarity_matrix.mean().mean()
            avg_accuracies.append(avg_similarity)

            cluster_id = "[" + "-".join(map(str, sorted([c.agent_id for c in cluster]))) + "]"
            try:
                self.metrics["clusters_accuracy"][cluster_id]['epoch'].append(epoch)
                self.metrics["clusters_accuracy"][cluster_id]['accuracy'].append(avg_similarity)
            except: 
                self.metrics["clusters_accuracy"][cluster_id] = {
                    'epoch': [epoch],
                    'accuracy': [avg_similarity]
                }
                
        avg_accuracy = sum(avg_accuracies) / len(avg_accuracies)
        try: self.metrics["accuracy"].append(avg_accuracy)
        except KeyError: self.metrics["accuracy"] = [avg_accuracy]

        clustering_percentage = 1 - len(self.clusters[0])/len(self.agents.keys()) if 0 in self.clusters else 1
        try: self.metrics["clustering_percentage"].append(clustering_percentage)
        except KeyError: self.metrics["clustering_percentage"] = [clustering_percentage]

        print(f"[DIFFUSER] Accuracy: {avg_accuracy:.4f} - Clustering percentage: {clustering_percentage:.4f}")


    def compute_similarity_matrix(self, embeddings, sample_size=None, plot=False):
        """
        Compute the similarity matrix of the agent embeddings.
        """
        keys = list(embeddings.keys())
        similarity_matrix = pd.DataFrame(0.0, index=keys, columns=keys)
        full_similarity_matrix = pd.DataFrame(0.0, index=keys, columns=keys)

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                agent_id_1 = keys[i]
                agent_id_2 = keys[j]
                embeddings_1 = embeddings[agent_id_1]
                embeddings_2 = embeddings[agent_id_2]
                similarity = 0

                if sample_size is not None:
                    for _ in range(sample_size):
                        index = np.random.randint(0, len(embeddings_1))
                        similarity += abs(cosine_similarity(embeddings_1[index], embeddings_2[index]))
                else:
                    for index in range(len(embeddings_1)):
                        similarity += abs(cosine_similarity(embeddings_1[index], embeddings_2[index]))

                similarity_matrix.loc[agent_id_1, agent_id_2] = similarity
                full_similarity_matrix.loc[agent_id_1, agent_id_2] = similarity
                full_similarity_matrix.loc[agent_id_2, agent_id_1] = similarity

        for agent_id in keys:
            full_similarity_matrix.loc[agent_id, agent_id] = sample_size

        similarity_matrix /= sample_size
        full_similarity_matrix /= sample_size

        plot and plot_similarity_matrix(full_similarity_matrix)
        return similarity_matrix, full_similarity_matrix
    

    def cluster_agents(self, embeddings, threshold=0.75, sample_size=5, plot=False):
        """
        Cluster the agents based on the similarity of their embeddings.
        """
        df_similarity, full_similarity_matrix = self.compute_similarity_matrix(embeddings, sample_size, plot)
        clusters = {
            0: []
        }
        agent_assignments = {}
        cluster_id = 1

        while True:
            if len(agent_assignments.keys()) == len(df_similarity):
                break

            max_similarity = df_similarity.max(numeric_only=True).max()
            if max_similarity < threshold:
                break

            agent_1, agent_2 = df_similarity.stack().idxmax()
            if agent_1 not in agent_assignments.keys() and agent_2 not in agent_assignments.keys():
                clusters[cluster_id] = [self.agents[agent_1], self.agents[agent_2]]
                agent_assignments[agent_1] = cluster_id
                agent_assignments[agent_2] = cluster_id
                cluster_id += 1
            elif agent_1 in agent_assignments.keys() and agent_2 not in agent_assignments.keys():
                clusters[agent_assignments[agent_1]].append(self.agents[agent_2])
                agent_assignments[agent_2] = agent_assignments[agent_1]
            elif agent_2 in agent_assignments.keys() and agent_1 not in agent_assignments.keys():
                clusters[agent_assignments[agent_2]].append(self.agents[agent_1])
                agent_assignments[agent_1] = agent_assignments[agent_2]

            df_similarity.loc[agent_1, agent_2] = 0

        for agent in self.agents.values():
            if agent.agent_id not in agent_assignments.keys():
                clusters[0].append(agent)
        if not clusters[0]:
            del clusters[0]

        return clusters, full_similarity_matrix


    def evaluate(self, test_datasets):
        """
        Evaluate the agent embeddings.
        """
        evaluate_data = {}
        for agent_id, data in test_datasets.items():
            self.agents[agent_id].receiving_queue.put({
                'action': 'EVALUATE',
                'data': {"results": True}
            })
        for _, data in test_datasets.items():
            message = self.agents[agent_id].results_queue.get()
            if (message['action'] == 'EVALUATE_COMPLETED'):
                evaluate_data[message['metadata']['agent_id']] = message['data']
            else:
                print(f"[EVALUATE] Wrong action. Received: {data['action']}, expected: EVALUATE_COMPLETED")
        return evaluate_data


    def process_epoch(self, clustering=False, plot=False):
        """
        Process a single epoch of training.
        """
        for agent in self.agents.values():
                agent.receiving_queue.put({'action': 'TRAIN'})

        epoch_completed = all([x for x in self.wait_agents_messages('EPOCH_COMPLETED').values()])
        
        epoch_embeddings = {}
        while not epoch_completed:
            local_embeddings = self.wait_agents_messages('BATCH_EMBEDDING')
            if len(self.clusters.keys()) == 1:
                cluster_embeddings = [local_embeddings[key] for key in local_embeddings.keys()]
                self.global_embedding = torch.mean(torch.stack(cluster_embeddings), dim=0).to(self.device)
            for id, cluster in self.clusters.items():
                if id != 0:
                    cluster_embeddings = [local_embeddings[agent.agent_id] for agent in cluster]
                    cluster_embedding = torch.mean(torch.stack(cluster_embeddings), dim=0).to(self.device)
                    for agent in cluster:
                        agent.receiving_queue.put({'action': 'GLOBAL_SYNC', 'data': cluster_embedding})
                elif len(self.clusters.keys()) > 1:
                    for agent in cluster:
                        similar_agents = [c for c in self.agents.values() if self.similarity_matrix.loc[agent.agent_id, c.agent_id] > 0.7]
                        similar_agents = len(similar_agents) > 1 and similar_agents or cluster
                        cluster_embeddings = [local_embeddings[agent.agent_id] for agent in similar_agents]
                        cluster_embedding = torch.mean(torch.stack(cluster_embeddings), dim=0).to(self.device)
                        agent.receiving_queue.put({'action': 'GLOBAL_SYNC', 'data': cluster_embedding})
                else:
                    for agent in cluster:
                        agent.receiving_queue.put({'action': 'GLOBAL_SYNC', 'data': self.global_embedding})


            epoch_completed = all([x for x in self.wait_agents_messages('EPOCH_COMPLETED').values()])
            for agent_id, embedding in local_embeddings.items():
                try: epoch_embeddings[agent_id].extend(embedding)
                except KeyError: epoch_embeddings[agent_id] = list(embedding)

        epoch_losses = [x for x in self.wait_agents_messages('EPOCH_LOSS').values()]
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.metrics["training_losses"].append(avg_loss)

        if self.clustering & clustering:
            clusters, similarity_matrix = self.cluster_agents(epoch_embeddings, 0.97, 50, plot=plot)
            self.clusters = clusters
            self.similarity_matrix = similarity_matrix
            print_cluster_info(self.clusters)

        return avg_loss
            


    def run(self, evaluate=True, plot=False):
        """
        Main training loop for diffuser.
        """
        self.initialize_global_embedding()
        self.start_agents()

        for epoch in range(self.epochs):
            print(f"[DIFFUSER] Epoch {epoch + 1}/{self.epochs}")
            avg_loss = self.process_epoch(epoch % self.clustering_step == self.clustering_step-1, plot=plot)
            print(f"[DIFFUSER] Epoch completed. Epoch loss {avg_loss:.8f}")
            if evaluate:
                self.compute_accuracy(epoch)

        return self.local_embeddings, self.metrics
