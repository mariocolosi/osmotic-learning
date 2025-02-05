# v0.1.0

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from simulator.misc.utils import set_seed


class Agent:
    def __init__(
            self, 
            agent_id, 
            model, 
            train_loader, 
            test_loader, 
            receiving_queue, 
            sending_queue, 
            results_queue = None, 
            device='cpu', 
            seed=42
        ):
        """
        Initialize the agent with model, data, queues, and device.
        """
        self.agent_id = agent_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.receiving_queue = receiving_queue
        self.sending_queue = sending_queue
        self.results_queue = results_queue
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device
        self.temperature = 0.1
        set_seed(seed)


    def get_embedding_shape(self):
        """
        Retrieve the output shape of the final layer.
        """
        output_shape = None
        try:
            first_layer = list(self.model.children())[0]
            input_dim = getattr(first_layer, 'input_size', getattr(first_layer, 'in_features', 1))
            dummy_input = torch.zeros((1, 5, input_dim)).to(self.device)
        except Exception as e:
            print(f"[AGENT] Failed to retrieve input dimension dynamically: {e}")
            dummy_input = torch.zeros((1, 5, 1)).to(self.device)

        def hook(module, input, output):
            nonlocal output_shape
            output_shape = output.shape

        last_layer = next(
            (layer for layer in reversed(list(self.model.modules()))
            if isinstance(layer, (nn.Linear, nn.GRU, nn.LSTM))),
            None
        )

        if last_layer is None:
            raise ValueError("[AGENT] No suitable final layer found (Linear, GRU, LSTM).")

        hook_handle = last_layer.register_forward_hook(hook)

        try:
            with torch.no_grad():
                self.model(dummy_input)
        except Exception as e:
            print(f"[AGENT] Forward pass failed: {e}")
            raise RuntimeError("[AGENT] Unable to determine output shape with the dummy input.")

        hook_handle.remove()
        if output_shape is None:
            raise RuntimeError("[AGENT] The forward hook failed to capture the output shape of the final layer.")

        return output_shape
        

    def forward_pass(self, data):
        """
        Perform the forward pass and return the local embedding.
        """
        data = data.to(self.device)
        return self.model(data)


    def calculate_alignment_loss(self, local_embedding, global_embedding):
        """
        Calculate alignment loss between local and global embeddings.
        """
        min_size = min(local_embedding.size(0), global_embedding.size(0))
        local_embedding = local_embedding[:min_size]
        global_embedding = global_embedding[:min_size]
        
        global_embedding = global_embedding.to(self.device)
        return self.loss_function(local_embedding, global_embedding)


    def calculate_mutual_information_loss(self, local_embedding, data):
        """
        Calculate an approximation of mutual information using contrastive loss.
        """
        batch_size = local_embedding.size(0)
        negative_samples = torch.randn_like(local_embedding)

        positive_similarity = F.cosine_similarity(local_embedding, self.model(data), dim=-1)
        negative_similarity = F.cosine_similarity(local_embedding, negative_samples, dim=-1)

        labels = torch.zeros(batch_size).to(local_embedding.device)
        logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity.unsqueeze(1)], dim=1)
        mi_loss = F.cross_entropy(logits / self.temperature, labels.long())

        return mi_loss
    

    def backward_pass(self, loss):
        """
        Perform backpropagation.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train_batch(self, data):
        """
        Train on a single batch with communication to the diffuser.
        """
        # Step 1: Forward Pass
        local_embedding = self.forward_pass(data)

        # Step 2: Send Local Embedding to Diffuser
        self.sending_queue.put({
            'action': 'BATCH_EMBEDDING',
            'data': local_embedding.detach().cpu(),
            'metadata': {'agent_id': self.agent_id}
        })

        # Step 3: Wait for Global Embedding
        while True:
            message = self.receiving_queue.get()
            if message['action'] == 'GLOBAL_SYNC':
                global_embedding = message['data']
                break

        # Step 4: Alignment Loss and Backpropagation
        alignment_loss = self.calculate_alignment_loss(local_embedding, global_embedding)

        # Step 5: Mutual Information Loss
        mi_loss = self.calculate_mutual_information_loss(local_embedding, data)

        # Step 6: Combined Loss
        lambda_alignment = 0.5
        lambda_mi = 0.5
        total_loss = lambda_alignment * alignment_loss + lambda_mi * mi_loss

        # Step 7: Backpropagation
        self.backward_pass(total_loss)

        return total_loss.item()


    def evaluate(self, dataloader):
        """
        Evaluate the model on the test set.
        """
        outputs = []
        self.model.eval() 
        with torch.no_grad():
            for batch in dataloader:
                batch_outputs = self.forward_pass(batch[0])
                outputs.extend(batch_outputs.detach().cpu())
        return outputs
    

    def run(self):
        """
        Main loop processing tasks from the task queue.
        """
        while True:
            message = self.receiving_queue.get()
            action = message.get('action')

            if action == 'STOP':
                break

            if action == 'TRAIN':
                total_loss = 0
                num_batches = 0

                for data in self.train_loader:
                    self.sending_queue.put({
                        'action': 'EPOCH_COMPLETED',
                        'data': False,
                        'metadata': {'agent_id': self.agent_id}
                    })
                    loss = self.train_batch(data[0])
                    total_loss += loss
                    num_batches += 1

                self.sending_queue.put({
                    'action': 'EPOCH_COMPLETED',
                    'data': True,
                    'metadata': {'agent_id': self.agent_id}
                })

                avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                self.sending_queue.put({
                    'action': 'EPOCH_LOSS',
                    'data': avg_loss,
                    'metadata': {'agent_id': self.agent_id, 'status': 'COMPLETED'}
                })

            if action == 'EVALUATE':
                outputs = self.evaluate(self.test_loader)
                sending_queue = self.results_queue if 'data' in message and message['data']['results'] else self.sending_queue
                if(sending_queue is not None):
                    sending_queue.put({
                        'action': 'EVALUATE_COMPLETED',
                        'data': outputs,
                        'metadata': {'agent_id': self.agent_id}
                    })