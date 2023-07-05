import torch
import random
import torch.nn as nn


class GRU4Rec(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_dim,
        item_num,
        state_size,
        action_dim,
        gru_layers=1,
        use_packed_seq=True,
        train_pad_embed=True,
    ):
        super(GRU4Rec, self).__init__()

        self.layers = gru_layers
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.item_num = item_num
        self.state_size = state_size
        self.action_dim = action_dim
        self.use_packed_seq = use_packed_seq

        # Item-embeddings
        padding_idx = self.item_num
        self.embedding = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=None if train_pad_embed else padding_idx,
        )

        # Init strategy like in paper
        self.embedding.weight.data.normal_(mean=0, std=0.01)

        # Set padding embedding back to zero after init if its untrainable
        if not train_pad_embed:
            with torch.no_grad():
                self.embedding.weight[padding_idx] = torch.zeros(self.embedding_dim)

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layers,
            bias=True,
            batch_first=True,
        )

        # Output layer
        self.output = nn.Linear(
            in_features=self.hidden_size, out_features=self.action_dim
        )

    def forward(self, s, lengths):
        # s - (batch_size, n_memory)
        # lengths - (batch_size, true_state_len)
        # out - (batch_size, actions)

        # Use packed sequences:
        if self.use_packed_seq:
            emb_seq = self.embedding(s)
            emb_seq = nn.utils.rnn.pack_padded_sequence(
                emb_seq, lengths=lengths, batch_first=True, enforce_sorted=False
            )
        else:
            emb_seq = self.embedding(s)  # (batch_size, n_memory, embedding_dim)

        _, h = self.gru(emb_seq)
        h = h[0, :, :]  # Output is (D*num_layers, batch, hidden)

        # Output logits
        out = self.output(h)

        return out


class GRU4Rec_trainer:
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        layers,
        train_pad_embed,
        use_packed_seq,
        learning_rate,
        item_num,
        state_size,
        action_dim,
        device,
        action_types,
        action_types_dict,
        torch_rand_seed=118,
        python_rand_seed=999,
    ):
        # Set seeds
        torch.manual_seed(torch_rand_seed)
        random.seed(python_rand_seed)

        # Init both networks
        self.gru_model = GRU4Rec(
            hidden_size=hidden_dim,
            embedding_dim=embedding_dim,
            train_pad_embed=train_pad_embed,
            use_packed_seq=use_packed_seq,
            item_num=item_num,
            state_size=state_size,
            action_dim=action_dim,
            gru_layers=layers,
        )

        self.device = device
        self.learning_rate = learning_rate
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=None, reduction="mean")

        # Store unique action types (important for evaluation stats)
        self.action_types = action_types
        self.action_types_dict = action_types_dict

        # Init optimizers, one for each network
        self.optimizer = torch.optim.Adam(
            self.gru_model.parameters(),
            lr=self.learning_rate,
        )

    def train_step(self, s, a, true_len):
        """
        Performs one training step.
        s - (batch_size, n_memory)
        a - (batch_size) index of item/ID
        true_len - (batch_size)
        """

        # Forward pass

        # Transfer data to device
        s = s.to(self.device)
        a = a.to(self.device)

        preds = self.gru_model(s, true_len)

        # Supervised Loss between actual action and supervised prediction
        sup_loss = self.cross_entropy_loss(preds, a)  # a is index of action

        # Compute gradients
        self.optimizer.zero_grad()
        sup_loss.backward()

        # Take learning step
        self.optimizer.step()

        return sup_loss.item()

    def set_train(self):
        """
        Sets model in training mode
        """
        self.gru_model.train()

    def set_eval(self):
        """
        Sets models in evaluation mode
        """
        self.gru_model.eval()

    def send_to_device(self):
        """
        Send model to specified device.
        """
        self.gru_model.to(self.device)
