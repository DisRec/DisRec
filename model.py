import torch
import torch.nn as nn
import numpy as np


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda(3)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class DisRec(nn.Module):
    def __init__(self, config, gi_dataset, ui_dataset, device):
        super(DisRec, self).__init__()
        self.u_emb_dim = config.u_emb_size
        self.g_emb_dim = config.g_emb_size
        self.layers = config.layers
        self.drop_ratio = config.drop_ratio

        self.num_users = gi_dataset.num_users
        self.num_items = gi_dataset.num_items
        self.num_groups = gi_dataset.num_groups
        self.group_member_dict = gi_dataset.group_member_dict

        adj, D, A = gi_dataset.adj, gi_dataset.D, gi_dataset.A
        D = torch.Tensor(D).to(device)
        A = torch.Tensor(A).to(device)
        values = adj.data
        indices = np.vstack(
            (adj.row, adj.col))
        i = torch.LongTensor(indices).to(device)
        v = torch.FloatTensor(values).to(device)
        shape = adj.shape
        adj = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)

        self.adj = adj
        self.D = D
        self.A = A

        self.user_embedding = nn.Embedding(self.num_users, self.u_emb_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.u_emb_dim)
        self.group_embedding = nn.Embedding(self.num_groups, self.g_emb_dim)
        self.hyper_graph = HyperConv(self.layers)
        self.group_graph = GroupConv(self.layers)
        self.attention1 = AttentionLayer(2 * self.u_emb_dim, self.drop_ratio)
        self.attention2 = AttentionLayer(2 * self.g_emb_dim, self.drop_ratio)
        self.predict1 = PredictLayer(3 * self.u_emb_dim, self.drop_ratio)
        self.predict2 = PredictLayer(3 * self.g_emb_dim, self.drop_ratio)

        self.ui_dataset = ui_dataset
        self.device = device
        self.preference = PreferenceLayer(self.ui_dataset, self.u_emb_dim, self.layers)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.group_embedding.weight)

        self.fc_layer = torch.nn.Linear(self.g_emb_dim, self.g_emb_dim, bias=True)
        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.zeros_(self.fc_layer.bias)

        self.bilinear_layer = nn.Bilinear(self.g_emb_dim, self.g_emb_dim, 1)  # output_dim = 1 => single score.
        nn.init.zeros_(self.bilinear_layer.weight)
        nn.init.zeros_(self.bilinear_layer.bias)

        self.f = nn.ReLU()

        self.att_gate = nn.Sequential(nn.Linear(self.g_emb_dim, 1), nn.Sigmoid())
        self.hyper_gate = nn.Sequential(nn.Linear(self.g_emb_dim, 1), nn.Sigmoid())

    def reg_loss(self, group, pos, neg):
        group_emb_ego = self.group_embedding(group)
        pos_emb_ego = self.item_embedding(pos)
        neg_emb_ego = self.item_embedding(neg)
        reg_loss = (1 / 2) * (group_emb_ego.norm(2).pow(2) +
                              pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2)) / float(len(group))
        return reg_loss

    def forward(self, group_inputs, user_inputs, item_inputs, friends, friends_mask, members, members_mask):
        if (group_inputs is not None) and (user_inputs is None):
            u_p_emb, i_p_emb = self.preference.embedding_user.weight, self.preference.embedding_item.weight
            u_s_emb, i_s_emb = self.user_embedding.weight, self.item_embedding.weight
            user_embedding = torch.cat((u_p_emb, u_s_emb), dim=1)
            item_embedding = torch.cat((i_p_emb, i_s_emb), dim=1)
            item_emb = item_embedding[item_inputs]

            group_embedding = self.group_graph(self.group_embedding.weight, self.D, self.A)

            bsz = group_inputs.shape[0]
            max_len = members_mask.shape[1]
            members_emb = user_embedding[members]

            # attention aggregation
            item_emb_attn = item_emb.unsqueeze(1).expand(bsz, max_len, -1)
            at_emb = torch.cat((members_emb, item_emb_attn), dim=2)
            at_wt = self.attention2(at_emb, members_mask)
            g_emb_with_attention = torch.matmul(at_wt.unsqueeze(1), members_emb).squeeze()

            if self.training is True:
                # ssl
                max_at = at_wt.clone()
                max_indices = torch.argmax(at_wt, dim=1)
                for i, max_index in enumerate(max_indices):
                    max_at[i, max_index] = 0
                min_at = at_wt.clone()
                for i, row in enumerate(at_wt):
                    nonzero_indices = torch.nonzero(row).squeeze()
                    nonzero_values = row[nonzero_indices]
                    min_index = torch.argmin(nonzero_values).item()
                    min_at[i, min_index] = 0
                g_without_max = torch.matmul(max_at.unsqueeze(1), members_emb).squeeze()
                g_without_min = torch.matmul(min_at.unsqueeze(1), members_emb).squeeze()
                anchor = self.fc_layer(g_emb_with_attention)
                anchor = torch.tanh(anchor)
                pos = self.fc_layer(g_without_min)
                pos = torch.tanh(pos)
                neg = self.fc_layer(g_without_max)
                neg = torch.tanh(neg)
                distance_pos = torch.pairwise_distance(anchor, pos, p=2)
                distance_neg = torch.pairwise_distance(anchor, neg, p=2)
                y2 = self.f(distance_pos - distance_neg + 1.0)

                g_emb_pure = group_embedding[group_inputs]
                att_coef, hyper_coef = self.att_gate(g_emb_with_attention), self.hyper_gate(
                    g_emb_pure)
                group_emb = att_coef * g_emb_with_attention + hyper_coef * g_emb_pure

                element_emb = torch.mul(group_emb, item_emb)
                new_emb = torch.cat((element_emb, group_emb, item_emb), dim=1)
                y = torch.sigmoid(self.predict2(new_emb))
                return y, y2
            else:
                g_emb_pure = group_embedding[group_inputs]
                group_emb = g_emb_with_attention + g_emb_pure
                element_emb = torch.mul(group_emb, item_emb)
                new_emb = torch.cat((element_emb, group_emb, item_emb), dim=1)
                y = torch.sigmoid(self.predict2(new_emb))
                return y

        else:
            ui_embedding = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
            ui_embedding = self.hyper_graph(self.adj, ui_embedding)
            user_embedding, item_embedding = torch.split(ui_embedding, [self.num_users, self.num_items], dim=0)
            user_emb = user_embedding[user_inputs]
            item_emb = item_embedding[item_inputs]

            bsz = user_inputs.shape[0]
            max_len = friends_mask.shape[1]
            friends_emb = user_embedding[friends]

            user_emb_attn = user_emb.unsqueeze(1).expand(bsz, max_len, -1)
            at_emb = torch.cat((friends_emb, user_emb_attn), dim=2)
            at_wt = self.attention1(at_emb, friends_mask)
            user_emb_with_attention = torch.matmul(at_wt.unsqueeze(1), friends_emb).squeeze()

            user_emb_pure = user_embedding[user_inputs]
            user_emb = user_emb_with_attention + user_emb_pure

            element_emb = torch.mul(user_emb, item_emb)
            new_emb = torch.cat((element_emb, user_emb, item_emb), dim=1)
            y = torch.sigmoid(self.predict1(new_emb))
            return y


class HyperConv(nn.Module):
    def __init__(self, layers):
        super(HyperConv, self).__init__()
        self.layers = layers

    def forward(self, adj, embedding):
        all_emb = embedding
        final = [all_emb]
        for i in range(self.layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            final.append(all_emb)
        final_emb = torch.sum(torch.stack(final), dim=0)
        return final_emb


class GroupConv(nn.Module):
    def __init__(self, layers):
        super(GroupConv, self).__init__()
        self.layers = layers

    def forward(self, embedding, D, A):
        DA = torch.mm(D, A).float()
        group_emb = embedding
        final = [group_emb]
        for i in range(self.layers):
            group_emb = torch.mm(DA, group_emb)
            final.append(group_emb)
        final_emb = torch.sum(torch.stack(final), dim=0)
        return final_emb


class AttentionLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim / 2)),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(int(emb_dim / 2), 1)
        )

    def forward(self, x, mask):
        bsz = x.shape[0]
        out = self.linear(x)
        out = out.view(bsz, -1)  # [bsz, max_len]
        out.masked_fill_(mask.bool(), -np.inf)
        weight = torch.softmax(out, dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class PreferenceLayer(nn.Module):
    def __init__(self, dataset, emb_dim, layer):
        super(PreferenceLayer, self).__init__()
        self.emb_dim = emb_dim
        self.layer = layer
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.latent_dim = self.emb_dim
        self.n_layers = self.layer
        self.A_split = False
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        g = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g)):
                    temp_emb.append(torch.sparse.mm(g[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
