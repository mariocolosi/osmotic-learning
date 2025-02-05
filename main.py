# %% [markdown]
# # Osmotic Learning Experiments
# 
# This notebook implements and analyzes several experiments related to the **Osmotic Learning (OSM-L) paradigm**. 
# OSM-L is a self-supervised learning method that enables distributed entities to learn contextual representations without explicit supervision. 
# These experiments aim to validate the latent representation alignment, robustness to noise, and the emergence of dynamic sub-contexts.

# %%
from simulator.core.agent import Agent
from simulator.core.diffuser import Diffuser
from simulator.misc.utils import set_seed
from simulator.misc.visualization import plot_training_loss, plot_accuracy, plot_embeddings_similarity_matrix

from models.base import MODEL_REGISTRY
from torch.utils.data import DataLoader, TensorDataset
from datasets.DataProcessor import DataProcessor
import multiprocessing as mp

# %% [markdown]
# ## Experiment 1: Latent Representation and Training Convergence
# 
# In this experiment, we train two agents (Agent0 and Agent1) in a **simple context**, where both agents have correlated data. 
# The objective is to evaluate whether **local embeddings converge to a shared representation** using OSM-L's diffusion mechanism.
# 
# - **Key Hypothesis:** The embeddings of both agents should align, despite training occurring independently.
# - **Metric:** Similarity matrix of embeddings over time.
# - **Expected Outcome:** If OSM-L works correctly, embeddings generated by Agent0 and Agent1 at the same logical step should be highly similar.
# 

# %%
# Define parameters for the experiment

WINDOW_SIZE = 10
TRAIN_DATA_CSV = 'datasets/simple-context/train_data.csv'
TEST_DATA_CSV = 'datasets/simple-context/test_data.csv'
EPOCHS = 5
EMBEDDING_SIZE = 5
BATCH_SIZE = 50
CLUSTERING_STEP = 2
GRU_HIDDEN_DIM = 20
DEVICE = 'cpu' # 'cuda' or 'cpu' or 'mps'

set_seed(42)

# %% [markdown]
# ### Load and preprocess the dataset
# The dataset is loaded from CSV files, and only the first two agents (0 and 1) are retained.

# %%
dataProcessor = DataProcessor(window_size=WINDOW_SIZE)
data = dataProcessor.load_data_from_csv(TRAIN_DATA_CSV, TEST_DATA_CSV)

# Maintain only first two Agents from Simple Context
data = {
    'train': {
        0: data['train']['0'],
        1: data['train']['1']
    },
    'test': {
        0: data['test']['0'],
        1: data['test']['1']
    }
}
data = dataProcessor.load_data(data['train'], data['test'])
agents_data = dataProcessor.preprocess_data()

# %% [markdown]
# ### Plot time series
# This visualization helps verify the structure and correlation of the data.

# %%
dataProcessor.plot_time_series()

# %% [markdown]
# ### Initialize agents and communication queues
# Each agent is assigned its own queue for sending and receiving messages and processes local data using a GRU-based model.

# %%
NUM_AGENTS = len(agents_data['train'])
sending_queues = {agent_id: mp.Queue() for agent_id in agents_data['train'].keys()}
receiving_queues = {agent_id: mp.Queue() for agent_id in agents_data['train'].keys()}
results_queue = mp.Queue()

agents = {}
for agent_id, data in agents_data['train'].items():
    dataset = TensorDataset(data['series'])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataset = TensorDataset(agents_data['test'][agent_id]['series'])
    testloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = MODEL_REGISTRY["gru"](
        input_dim=data["features"],
        hidden_dim=GRU_HIDDEN_DIM,
        output_dim=EMBEDDING_SIZE
    )
    agent = Agent(
        agent_id=agent_id,
        model=model,
        train_loader=dataloader,
        test_loader=testloader,
        results_queue=results_queue,
        sending_queue=sending_queues[agent_id],
        receiving_queue=receiving_queues[agent_id],
        device=DEVICE
    )
    agents[agent_id] = agent


# %% [markdown]
# ### Initialize the Diffuser
# The **Diffuser** coordinates the learning process by aligning embeddings across agents.

# %%
diffuser = Diffuser(
    agents=agents,
    epochs=EPOCHS,
    clustering=True,
    clustering_step=CLUSTERING_STEP,
    device=DEVICE,
)

# %% [markdown]
# ### Run the training process
# This step allows **local embeddings to converge** while preserving individual data characteristics.

# %%
local_embeddings, metrics = diffuser.run(evaluate=True, plot=False)

# %% [markdown]
# ### Evaluate the model on the test set
# The evaluation measures how well the learned embeddings generalize.

# %%
evaluate_data = diffuser.evaluate(agents_data['test'])

# %% [markdown]
# ### Plot the embeddings similarity matrix
# This matrix visualizes how well the embeddings from different agents align over time.

# %%
plot_embeddings_similarity_matrix(evaluate_data, (6,4), tickevery=30, min=0.4, max=1, cbar=True)

# %% [markdown]
# **Results for Experiment 1: Latent Representation and Training Convergence**
# 
# The results confirm that **OSM-L successfully aligns local embeddings**, achieving a **0.9984 accuracy at epoch 4**. The similarity matrix shows that embeddings produced by Agent0 and Agent1 at the same logical step **converge to a shared representation**, despite independent training. Notably, in the interval [50, 100], the model correctly identifies a stationary phase, demonstrating its ability to **capture contextual consistency** even when local data exhibit oscillatory behavior.
# 

# %% [markdown]
# ### Stop the agents
# After training and evaluation, the agents are stopped to free up resources.

# %%
diffuser.stop_agents()

# %% [markdown]
# ## Experiment 2: Robustness of Learning
#  
# This experiment introduces **two misleading agents** (AgentM0 and AgentM1) with randomly generated data.
# The goal is to evaluate OSM-L’s **resilience against noise** and assess whether the diffusion process can **preserve alignment** among the correlated agents while ignoring uncorrelated ones.
# 
# - **Key Hypothesis:** The embeddings of correlated agents (Agent0 and Agent1) should remain aligned, while misleading agents should not influence the global representation.
# - **Metric:** Similarity matrices of all four agents.
# - **Expected Outcome:** The embeddings of misleading agents should not exhibit structured similarity patterns, confirming that OSM-L **filters out irrelevant information**.
# 

# %% [markdown]
# ### Reload and preprocess the dataset
# The same dataset is reloaded, this time without filtering any agents.

# %%
dataProcessor = DataProcessor(window_size=WINDOW_SIZE)
data = dataProcessor.load_data_from_csv(TRAIN_DATA_CSV, TEST_DATA_CSV)
data = dataProcessor.load_data(data['train'], data['test'])
agents_data = dataProcessor.preprocess_data()

# %% [markdown]
# ### Plot time series for all agents
# This visualization now includes the misleading agents.

# %%
dataProcessor.plot_time_series()

# %% [markdown]
# ### Initialize agents and queues
# All four agents (including the misleading ones) are included in this run.

# %%
NUM_AGENTS = len(agents_data['train'])
sending_queues = {agent_id: mp.Queue() for agent_id in agents_data['train'].keys()}
receiving_queues = {agent_id: mp.Queue() for agent_id in agents_data['train'].keys()}
results_queue = mp.Queue()

agents = {}
for agent_id, data in agents_data['train'].items():
    dataset = TensorDataset(data['series'])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataset = TensorDataset(agents_data['test'][agent_id]['series'])
    testloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = MODEL_REGISTRY["gru"](
        input_dim=data["features"],
        hidden_dim=GRU_HIDDEN_DIM,
        output_dim=EMBEDDING_SIZE
    )
    agent = Agent(
        agent_id=agent_id,
        model=model,
        train_loader=dataloader,
        test_loader=testloader,
        results_queue=results_queue,
        sending_queue=sending_queues[agent_id],
        receiving_queue=receiving_queues[agent_id],
        device=DEVICE
    )
    agents[agent_id] = agent

# %% [markdown]
# ### Initialize the Diffuser
# The **Diffuser** coordinates the learning process by aligning embeddings across agents.

# %%
diffuser = Diffuser(
    agents=agents,
    epochs=EPOCHS,
    clustering=True,
    clustering_step=CLUSTERING_STEP,
    device=DEVICE,
)

# %% [markdown]
# ### Run the training process
# This step allows **local embeddings to converge** while preserving individual data characteristics.

# %%
local_embeddings, metrics = diffuser.run(evaluate=True, plot=False)

# %% [markdown]
# ### Evaluate the model on the test set
# The evaluation measures how well the learned embeddings generalize.

# %%
evaluate_data = diffuser.evaluate(agents_data['test'])

# %% [markdown]
# ### Plot the embeddings similarity matrix
# This matrix visualizes how well the embeddings from different agents align over time.

# %%
plot_embeddings_similarity_matrix(evaluate_data, (6,4), tickevery=30, min=0.7, max=1, cbar=True)

# %% [markdown]
# **Results for Experiment 2: Robustness of Learning**
# 
# Even in the presence of misleading agents, OSM-L **preserves representation alignment** among the correlated agents (Agent0 and Agent1). The similarity matrix of these agents **retains its structured pattern**, whereas the misleading agents (AgentM0 and AgentM1) exhibit **randomized embeddings without a clear diagonal structure**. This confirms that **OSM-L effectively filters out uncorrelated information**, preventing the diffusion process from being influenced by noise.
# 

# %% [markdown]
# ### Stop the agents
# After training and evaluation, the agents are stopped to free up resources.

# %%
diffuser.stop_agents()

# %% [markdown]
# ## Experiment 3: Emergence of Sub-Contexts
# 
# In this experiment, we transition from a simple to a **complex context**, adding three more agents (Agent2, Agent3, and Agent4).
# The objective is to analyze how **OSM-L dynamically forms sub-contexts** among correlated agents.
# 
# - **Key Hypothesis:** OSM-L should **group agents into meaningful clusters** based on their correlation, without prior knowledge.
# - **Metric:** Evolution of similarity matrices over multiple epochs.
# - **Expected Outcome:** 
#     - Agents sharing strong correlations (Agent0 and Agent1; Agent2 and Agent3) should **form stable sub-contexts**.
#     - Agent4, which has **partial correlations** with other agents, should eventually align within an appropriate cluster.
# 
# The results will confirm whether OSM-L can **autonomously identify and adapt to latent structures in distributed data.**
# 

# %%
# Define parameters for the experiment

WINDOW_SIZE = 10
TRAIN_DATA_CSV = 'datasets/complex-context/train_data.csv'
TEST_DATA_CSV = 'datasets/complex-context/test_data.csv'
EPOCHS = 30
EMBEDDING_SIZE = 5
BATCH_SIZE = 50
CLUSTERING_STEP = 2
GRU_HIDDEN_DIM = 20
DEVICE = 'cpu' # 'cuda' or 'cpu' or 'mps'

set_seed(42)

# %%


# %% [markdown]
# ### Load and preprocess the dataset
# The dataset is loaded from CSV files from the 'complex context', consisting of Agent0, Agent1, Agent2, Agent3, Agent4.

# %%
dataProcessor = DataProcessor(window_size=WINDOW_SIZE)
data = dataProcessor.load_data_from_csv(TRAIN_DATA_CSV, TEST_DATA_CSV)
data = dataProcessor.load_data(data['train'], data['test'])
agents_data = dataProcessor.preprocess_data()

# %% [markdown]
# ### Plot time series for all agents
# This visualization now includes four agents.

# %%
dataProcessor.plot_time_series()

# %% [markdown]
# ### Initialize agents and communication queues
# Each agent is assigned its own queue for sending and receiving messages and processes local data using a GRU-based model.

# %%
NUM_AGENTS = len(agents_data['train'])
sending_queues = {agent_id: mp.Queue() for agent_id in agents_data['train'].keys()}
receiving_queues = {agent_id: mp.Queue() for agent_id in agents_data['train'].keys()}
results_queue = mp.Queue()

agents = {}
for agent_id, data in agents_data['train'].items():
    dataset = TensorDataset(data['series'])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataset = TensorDataset(agents_data['test'][agent_id]['series'])
    testloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = MODEL_REGISTRY["gru"](
        input_dim=data["features"],
        hidden_dim=GRU_HIDDEN_DIM,
        output_dim=EMBEDDING_SIZE
    )
    agent = Agent(
        agent_id=agent_id,
        model=model,
        train_loader=dataloader,
        test_loader=testloader,
        results_queue=results_queue,
        sending_queue=sending_queues[agent_id],
        receiving_queue=receiving_queues[agent_id],
        device=DEVICE
    )
    agents[agent_id] = agent

# %% [markdown]
# ### Initialize the Diffuser
# The **Diffuser** coordinates the learning process by aligning embeddings across agents.

# %%
diffuser = Diffuser(
    agents=agents,
    epochs=EPOCHS,
    clustering=True,
    clustering_step=CLUSTERING_STEP,
    device=DEVICE,
)

# %% [markdown]
# ### Run the training process
# This step allows **local embeddings to converge** while preserving individual data characteristics.

# %%
local_embeddings, metrics = diffuser.run(evaluate=True, plot=True)

# %% [markdown]
# ### Evaluate Training Loss and Accuracy over Epochs
# Plot training loss and accuracy showing their evolution during the training.

# %%
training_losses = metrics["training_losses"]
plot_training_loss(training_losses)
plot_accuracy(metrics)

# %% [markdown]
# ### Evaluate the model on the test set
# The evaluation measures how well the learned embeddings generalize.

# %%
evaluate_data = diffuser.evaluate(agents_data['test'])

# %% [markdown]
# ### Plot the embeddings similarity matrix
# This matrix visualizes how well the embeddings from different agents align over time.

# %%
plot_embeddings_similarity_matrix(evaluate_data, (6,4), tickevery=30, min=0.7, max=1, cbar=True)

# %% [markdown]
# **Results for Experiment 3: Emergence of Sub-Contexts**
# 
# The similarity matrices reveal the **formation of distinct sub-contexts**, with **Agent0-Agent1** and **Agent2-Agent4** converging into separate clusters. Despite a negative correlation, Agent2 and Agent3 **align within the same sub-context**, while Agent4 dynamically integrates based on its **partial correlation**. The training loss **rapidly decreases**, and accuracy peaks at **0.9954 at epoch 23**, confirming that OSM-L **dynamically structures contextual representations**, distinguishing relevant relationships without explicit supervision.
# 

# %% [markdown]
# ### Stop the agents
# After training and evaluation, the agents are stopped to free up resources.

# %%
diffuser.stop_agents()


