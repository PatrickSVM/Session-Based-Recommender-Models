import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class BidirGRU4Rec(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_dim,
        item_num,
        state_size,
        action_dim,
        gru_layers=1,
        dropout=0,
        use_packed_seq=True,
        train_pad_embed=True,
        padding_idx=None,
    ):
        super(BidirGRU4Rec, self).__init__()

        self.layers = gru_layers
        self.hidden_dim = hidden_size
        self.embedding_dim = embedding_dim
        self.item_num = item_num
        self.state_size = state_size
        self.action_dim = action_dim
        self.use_packed_seq = use_packed_seq
        self.gru_layers = gru_layers

        # Use item num as default padding idx
        if padding_idx == None:
            padding_idx = self.item_num

        # Item-embeddings
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
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            bidirectional=True,
            bias=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout)

        # Linear layer to produce summary of src sentence
        # Applied to last hidden state of each dirction
        # self.src_summary = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Output layer
        self.output = nn.Linear(
            in_features=self.hidden_dim * 2, out_features=self.action_dim
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

        # h (2, batch_size, 2*hidden_dim)
        _, h = self.gru(emb_seq)

        # Concatenate hidden state of the two directions on dim=1
        # (batch_size, 2*enc_hidden_dim)
        h = torch.cat([h[0, :, :], h[1, :, :]], dim=1)

        # Dropout
        h = self.dropout(h)

        # src_summary = self.src_summary(h)
        # src_summary = F.tanh(src_summary)

        # Output logits
        out = self.output(h)

        return out


class BidirGRU4Rec_trainer:
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        gru_layers,
        dropout,
        train_pad_embed,
        use_packed_seq,
        learning_rate,
        item_num,
        state_size,
        action_dim,
        device,
        padding_idx=None,
        torch_rand_seed=118,
        python_rand_seed=999,
    ):
        # Set seeds
        torch.manual_seed(torch_rand_seed)
        random.seed(python_rand_seed)

        # Init both networks
        self.gru_model = BidirGRU4Rec(
            hidden_size=hidden_dim,
            embedding_dim=embedding_dim,
            train_pad_embed=train_pad_embed,
            use_packed_seq=use_packed_seq,
            item_num=item_num,
            state_size=state_size,
            action_dim=action_dim,
            gru_layers=gru_layers,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        self.device = device
        self.learning_rate = learning_rate
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=None, reduction="mean")

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
