import torch
import random
import torch.nn as nn


# TODO: Names, etc to GRU and function that calls init_gru_sqn()
#       if model is GRU, ...


class SQN_Network(nn.Module):
    def __init__(
        self,
        hidden_dim,
        item_num,
        state_size,
        action_dim,
        gamma,
        gru_layers,
        embedding_dim=50,
        random_embed_init=True,
        train_pad_embed=True,
        use_packed_seq=False,
        padding_idx=None,
        name="DQNetwork",
    ):
        super(SQN_Network, self).__init__()
        self.state_size = state_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.item_num = int(item_num)
        self.random_embed_init = random_embed_init
        self.use_packed_seq = use_packed_seq
        self.action_dim = action_dim
        self.gru_layers = gru_layers
        self.gamma = gamma
        self.name = name

        # Item-Embeddings
        if random_embed_init:
            # Use item num as default padding idx
            if padding_idx == None:
                padding_idx = self.item_num

            # Set train_pad_embed to True if we use packed seqs (just safety,
            # will be disregarded anyways because of masking through packing)
            if use_packed_seq:
                train_pad_embed = True

            # Set padding item as trainable/non trainable depending on train_pad_embed
            self.embedding = nn.Embedding(
                num_embeddings=self.item_num + 1,
                embedding_dim=self.embedding_dim,
                padding_idx=None if train_pad_embed else padding_idx,
            )

            # Init strategy like in paper
            self.embedding.weight.data.normal_(mean=0, std=0.01)

            # Set padding embedding back to zero after init if its untrainable
            if not train_pad_embed:
                with torch.no_grad():
                    self.embedding.weight[padding_idx] = torch.zeros(self.embedding_dim)

        else:
            raise NotImplementedError("TODO: Pretrained embedings.")

        # Input size:  (batch_size, sequence_length, input_features)
        # Output size: (layers, batch_size, hidden_dim)
        self.base_model = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=gru_layers,
            bias=True,
            batch_first=True,
        )

        # Supervised Head
        self.sup_head_output = nn.Linear(
            in_features=self.hidden_dim, out_features=self.action_dim
        )

        # Q-Head
        self.q_head_output = nn.Linear(
            in_features=self.hidden_dim, out_features=self.action_dim
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

        _, h = self.base_model(
            emb_seq
        )  # Only safe hidden states (batch_size, hidden_dim)
        h = h[0, :, :]  # Output is (D*num_layers, batch, hidden)

        # Supervised Head
        out_sup = self.sup_head_output(h)

        # Q-Head
        out_q = self.q_head_output(h)

        return out_sup, out_q


class SQN_trainer:
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        train_pad_embed,
        use_packed_seq,
        learning_rate,
        item_num,
        state_size,
        action_dim,
        gamma,
        gru_layers,
        device,
        padding_idx=None,
        torch_rand_seed=118,
        python_rand_seed=999,
        name_1="SQN_1",
        name_2="SQN_2",
    ):
        # Set seeds
        torch.manual_seed(torch_rand_seed)
        random.seed(python_rand_seed)

        # Init both networks
        self.DQN_1 = SQN_Network(
            hidden_dim=hidden_dim,
            item_num=item_num,
            state_size=state_size,
            action_dim=action_dim,
            gamma=gamma,
            gru_layers=gru_layers,
            embedding_dim=embedding_dim,
            train_pad_embed=train_pad_embed,
            use_packed_seq=use_packed_seq,
            padding_idx=padding_idx,
            name=name_1,
        )
        self.DQN_2 = SQN_Network(
            hidden_dim=hidden_dim,
            item_num=item_num,
            state_size=state_size,
            action_dim=action_dim,
            gamma=gamma,
            gru_layers=gru_layers,
            embedding_dim=embedding_dim,
            train_pad_embed=train_pad_embed,
            use_packed_seq=use_packed_seq,
            padding_idx=padding_idx,
            name=name_2,
        )

        self.device = device
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=None, reduction="mean")

        # Init optimizers, one for each network
        self.optimizer_1 = torch.optim.Adam(
            self.DQN_1.parameters(),
            lr=self.learning_rate,
        )

        self.optimizer_2 = torch.optim.Adam(
            self.DQN_2.parameters(),
            lr=self.learning_rate,
        )

    def train_step(self, s, a, r, s_next, true_len, true_next_len, is_end):
        """
        Performs one training step using double Q-Learning.
        s - (batch_size, n_memory)
        a - (batch_size) index of item/ID
        r - (batch_size) contains immediate rewards for t
        s_next - (batch_size, n_memory)
        true_len - (batch_size)
        is_end - (batch_size)
        """

        # Forward pass

        # Unsqueeze r to fit shapes
        r = r.unsqueeze(1)

        # Transfer data to device
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s_next = s_next.to(self.device)

        # Decide on which network to use as main DQN,
        # and which one to use for bootstrapped Q-value
        n_rand = random.uniform(0, 1)
        if n_rand <= 0.5:
            DQN_main = self.DQN_1
            DQN_boot = self.DQN_2
            optimizer = self.optimizer_1

        else:
            DQN_main = self.DQN_2
            DQN_boot = self.DQN_1
            optimizer = self.optimizer_2

        sup, q_out = DQN_main(s, true_len)

        # Select Q(s,a) from full output according to action tensor (expects unsqueezed action tensor)
        a_idx = a.unsqueeze(1)
        q = q_out.gather(1, a_idx)

        # Compute Q_boot(s', max_act_Q_main) without gradient tracking
        with torch.no_grad():
            sup_out_next, q_out_next = DQN_main(s_next, true_next_len)

            # Choose max action q_out_main(s_t+1) for each s in batch (batch,actions) -> (batch)
            max_a_next = torch.argmax(q_out_next, dim=1).unsqueeze(1)

            _, q_out_next_boot = DQN_boot(s_next, true_len)
            q_next_boot = q_out_next_boot.gather(1, max_a_next)

            # Assign Q_boot(s', max_act_Q_main) if end state
            q_next_boot[is_end] = 0.0

        # Q-Loss
        q_loss = (r + self.gamma * q_next_boot - q) ** 2
        q_loss = torch.mean(q_loss)

        # Supervised Loss between actual action and supervised prediction
        sup_loss = self.cross_entropy_loss(sup, a)  # a is index of action

        # Total Loss
        loss_total = q_loss + sup_loss

        # Compute gradients
        optimizer.zero_grad()
        loss_total.backward()

        # Take learning step
        optimizer.step()

        return sup_loss.item(), q_loss.item()

    def set_train(self):
        """
        Sets models in training mode
        (only important if contians dropout/batchnorm/...)
        """
        self.DQN_1.train()
        self.DQN_2.train()

    def set_eval(self):
        """
        Sets models in evaluation mode
        (only important if contians dropout/batchnorm/...)
        """
        self.DQN_1.eval()
        self.DQN_2.eval()

    def send_to_device(self):
        """
        Send model to specified device.
        """
        self.DQN_1.to(self.device)
        self.DQN_2.to(self.device)


if __name__ == "__main__":
    pass
