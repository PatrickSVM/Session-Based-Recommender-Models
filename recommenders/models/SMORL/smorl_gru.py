import torch
import random
import torch.nn as nn
from recommenders.utils.tensor_operations import (
    gather_from_3d,
    weighted_q_loss,
    get_weighted_q_target,
    get_max_action,
)
from recommenders.evaluate.novelty import get_batch_novelty_rewards
from recommenders.evaluate.diversity import get_batch_diversity_rewards


class SMORL_GRU_Net(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        item_num,
        state_size,
        action_dim,
        q_weights,
        gamma,
        gru_layers=1,
        random_embed_init=True,
        train_pad_embed=True,
        use_packed_seq=False,
        padding_idx=None,
        name="SMORL_GRU_Network",
    ):
        super(SMORL_GRU_Net, self).__init__()
        self.state_size = state_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.item_num = int(item_num)
        self.random_embed_init = random_embed_init
        self.use_packed_seq = use_packed_seq
        self.action_dim = action_dim
        self.gamma = gamma
        self.q_weights = q_weights
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

        # GRU-Base-Model

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

        # Accuracy/Relevance Q-Head #1
        self.q_head_acc = nn.Linear(
            in_features=self.hidden_dim, out_features=self.action_dim
        )

        # Diversity Q-Head #2
        self.q_head_div = nn.Linear(
            in_features=self.hidden_dim, out_features=self.action_dim
        )

        # Novelty Q-Head #3
        self.q_head_nov = nn.Linear(
            in_features=self.hidden_dim, out_features=self.action_dim
        )

    def forward(self, s, lengths):
        # s - (batch_size, n_memory)
        # lengths - (batch_size)
        # out_sup - (batch_size, actions)
        # out_all_q - (batch_size, q_heads, actions)

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

        # Accuracy Q-Head
        out_q_acc = self.q_head_acc(h)

        # Diversity Q-Head
        out_q_div = self.q_head_div(h)

        # Novelty Q-Head
        out_q_nov = self.q_head_nov(h)

        # Stack into one tensor (batch_size, q-heads, action_dim)
        out_all_q = torch.stack([out_q_acc, out_q_div, out_q_nov], dim=1)

        return out_sup, out_all_q


class SMORL_trainer:
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        padding_pos,
        train_pad_embed,
        use_packed_seq,
        learning_rate,
        item_num,
        state_size,
        action_dim,
        gamma,
        gru_layers,
        q_weights,
        alpha,
        div_embedding,
        unpopular_actions_set,
        topk_div,
        topk_nov,
        nov_rew_sig,
        device,
        input_tokenizer=None,
        output_tokenizer=None,
        padding_idx=None,
        torch_rand_seed=118,
        python_rand_seed=999,
        name_1="SMORL_1",
        name_2="SMORL_2",
    ):
        # Set seeds
        torch.manual_seed(torch_rand_seed)
        random.seed(python_rand_seed)

        # Init both networks
        self.SMORL_1 = SMORL_GRU_Net(
            hidden_dim=hidden_dim,
            item_num=item_num,
            state_size=state_size,
            action_dim=action_dim,
            gru_layers=gru_layers,
            gamma=gamma,
            q_weights=q_weights,
            embedding_dim=embedding_dim,
            train_pad_embed=train_pad_embed,
            use_packed_seq=use_packed_seq,
            padding_idx=padding_idx,
            name=name_1,
        )
        self.SMORL_2 = SMORL_GRU_Net(
            hidden_dim=hidden_dim,
            item_num=item_num,
            state_size=state_size,
            action_dim=action_dim,
            gru_layers=gru_layers,
            q_weights=q_weights,
            gamma=gamma,
            embedding_dim=embedding_dim,
            train_pad_embed=train_pad_embed,
            use_packed_seq=use_packed_seq,
            padding_idx=padding_idx,
            name=name_2,
        )

        # Save parameters
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.q_weights = q_weights
        self.padding_pos = padding_pos
        self.learning_rate = learning_rate
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=None, reduction="mean")
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        # Save reward computation parameters
        self.div_embedding = div_embedding
        self.topk_div = topk_div
        self.unpopular_actions_set = unpopular_actions_set
        self.topk_nov = topk_nov
        self.nov_rew_signal = nov_rew_sig

        # Init optimizers, one for each network
        self.optimizer_1 = torch.optim.Adam(
            self.SMORL_1.parameters(),
            lr=self.learning_rate,
        )

        self.optimizer_2 = torch.optim.Adam(
            self.SMORL_2.parameters(),
            lr=self.learning_rate,
        )

    def train_step(self, s, a, r_acc, s_next, true_len, true_next_len, is_end):
        """
        Performs one training step using double Q-Learning.
        s - (batch_size, state_len)
        a - (batch_size) index of item/ID
        r_acc - (batch_size) contains offline reward for accuracy
        s_next - (batch_size, state_len)
        true_len - (batch_size)
        is_end - (batch_size)
        """

        # Forward pass

        # Unsqueeze r to fit shapes
        r_acc = r_acc.unsqueeze(1)

        # Transfer data to device (len is expected to be on cpu)
        s = s.to(self.device)
        a = a.to(self.device)
        r_acc = r_acc.to(self.device)
        s_next = s_next.to(self.device)
        is_end = is_end.to(self.device)

        # Decide on which network to use as main SMORL,
        # and which one to use for bootstrapped Q-values
        n_rand = random.uniform(0, 1)
        if n_rand <= 0.5:
            SMORL_main = self.SMORL_1
            SMORL_boot = self.SMORL_2
            optimizer = self.optimizer_1

        else:
            SMORL_main = self.SMORL_2
            SMORL_boot = self.SMORL_1
            optimizer = self.optimizer_2

        # Get outputs, q_all cols: acc, div, nov
        sup, q_all = SMORL_main(s, true_len)

        # Compute sup loss here - later sup on CPU! Faster here on GPU.
        sup_loss = self.cross_entropy_loss(sup, a)  # a is index of action

        # Get Q_i(s,a) for all heads i (batch_size, q_heads)
        q = gather_from_3d(q_all, a)

        # Compute target - Q_boot(s', max_act_Q_main) without gradient tracking
        with torch.no_grad():
            _, q_out_next = SMORL_main(s_next, true_next_len)

            # Get weighted Q for all actions in all samples (batch_size, action_dim)
            q_next = get_weighted_q_target(
                q_vals=q_out_next, w=self.q_weights.to(self.device)
            )

            # Choose a*: max action q_out_main(s_t+1) for each s in batch
            # (batch_size, actions) -> (batch_size)
            max_a_next = get_max_action(q_next)

            # Compute Q_boot(s', a*) (batch_size, q_heads)
            _, q_out_next_boot = SMORL_boot(s_next, true_len)
            q_next_boot = gather_from_3d(q_out_next_boot, max_a_next)

            # Assign Q_boot(s', max_act_Q_main)=0 if end state
            q_next_boot[is_end, :] = 0.0

            # Compute Diversity rewards for batch
            r_div = get_batch_diversity_rewards(
                s=s,
                predictions=sup,
                len_states=true_next_len,
                padding_pos=self.padding_pos,
                topk_to_consider=self.topk_div,
                embedding_layer=self.div_embedding,
                device=self.device,
                input_tokenizer=self.input_tokenizer,
                output_tokenizer=self.output_tokenizer,
            )

            # Compute Novelty rewards for batch - CPU (inside func)
            r_nov = get_batch_novelty_rewards(
                predictions=sup,
                unpopular_items=self.unpopular_actions_set,
                reward=self.nov_rew_signal,
                topk_to_consider=self.topk_nov,
            )

            r_nov = torch.Tensor(r_nov).to(self.device)

            # Get r - (batch_size, q_heads)
            r = torch.stack([r_acc.squeeze(), r_div, r_nov], dim=1)

            # Cast to float32, can be float64 sometimes
            r = r.to(torch.float32)

        # SDQL-Loss
        q_loss_all_heads = (r + self.gamma * q_next_boot - q) ** 2

        q_loss = weighted_q_loss(
            q_vals=q_loss_all_heads, w=self.q_weights.to(self.device)
        )
        q_loss = torch.mean(q_loss)

        # Total Loss
        loss_total = sup_loss + self.alpha * q_loss

        # Compute gradients
        optimizer.zero_grad()
        loss_total.backward()

        # Take learning step
        optimizer.step()

        return sup_loss.item(), q_loss.item()

    def set_train(self):
        """
        Sets models in training mode
        """
        self.SMORL_1.train()
        self.SMORL_2.train()

    def set_eval(self):
        """
        Sets models in evaluation mode
        """
        self.SMORL_1.eval()
        self.SMORL_2.eval()

    def send_to_device(self):
        """
        Send model to specified device.
        """
        self.SMORL_1.to(self.device)
        self.SMORL_2.to(self.device)


if __name__ == "__main__":
    pass
