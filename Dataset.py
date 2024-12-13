import scipy.sparse as sp
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from time import time
from torch.utils import data
from util import save, load


def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(" ")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList


def load_negative_file(filename, num_items):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(" ")
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            for j in range(num_items - len(negatives)):
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList


def load_rating_file_as_matrix(filename):
    # Get number of users and items
    num_users = 0
    num_items = 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(" ")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    # Construct matrix
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(" ")
            if len(arr) > 2:
                user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
            else:
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0
            line = f.readline()
    return mat

class UserItemDataset(data.Dataset):
    def __init__(self, path, num_negatives, social_num, device):
        self.device = device
        self.path = path
        self.num_negatives = num_negatives  # 10
        self.social_num = social_num
        # social data
        self.user_trainMatrix = load_rating_file_as_matrix(path + "userRatingTrain.txt")  # (0, 539)	1.0
        self.num_users, self.num_items = self.user_trainMatrix.shape
        self.friends, self.friends_mask = self.get_social(path + "social.txt", self.num_users, self.social_num)
        self.S = None
        # preference Data
        self.users_D = np.array(self.user_trainMatrix.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.user_trainMatrix.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self.Graph = None
        print("# Train User-Item Interaction:", self.user_trainMatrix.nnz * self.num_negatives)
        print("Finish loading U-I training dataset.")

    def __len__(self):
        return self.user_trainMatrix.nnz * self.num_negatives

    def __getitem__(self, index):
        user, pos, neg, friends, friends_mask = self.S[index]
        user = torch.from_numpy(np.array(user, dtype=np.int64))
        pos = torch.from_numpy(np.array(pos, dtype=np.int64))
        neg = torch.from_numpy(np.array(neg, dtype=np.int64))
        friends = torch.from_numpy(np.array(friends, dtype=np.int64))
        friends_mask = torch.from_numpy(np.array(friends_mask, dtype=np.int64))
        return user, pos, neg, friends, friends_mask

    def get_user_train_instances(self, train, friends, friends_mask):
        num_users = train.shape[0]
        num_items = train.shape[1]
        S = []
        for (u, i) in train.keys():
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                S.append([u, i, j, friends[u], friends_mask[u]])
        return S

    def get_social(self, path, num_users, social_num):
        u_f_d = {}
        u_mask_d = {}
        max_len = 0
        with open(path) as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(": ")
                user, item = int(arr[0]), [int(x) for x in arr[1].split(" ")]
                u_f_d[user] = item
                max_len = max(max_len, len(item))
                line = f.readline()
        max_len = min(max_len, social_num)
        for i in range(num_users):
            user = i
            if u_f_d.get(i) == None:
                follows = [i]
            else:
                follows = u_f_d[i]
            cur_len = len(follows)
            if cur_len > max_len:
                cur_len = max_len
                follows = follows[:max_len]
            follows_masked = np.append(follows, np.zeros(max_len - cur_len))
            mask = np.zeros(max_len)
            mask[cur_len:] = 1.0
            u_f_d[user] = follows_masked
            u_mask_d[user] = mask
        return u_f_d, u_mask_d

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading user preference embedding adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + 's_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating user preference embedding adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.user_trainMatrix.tolil()
                adj_mat[:self.num_users, self.num_users:] = R
                adj_mat[self.num_users:, :self.num_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + 's_pre_adj_mat.npz', norm_adj)

            self.Graph = norm_adj
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)
        return self.Graph

class UserEvalDataset(data.Dataset):
    def __init__(self, path, friends, friends_mask, num_items):
        self.user_testRatings = load_rating_file_as_list(path + "userRatingTest.txt")
        self.user_testNegatives = load_negative_file(path + "userRatingNegative.txt", num_items)
        self.friends, self.friends_mask = friends, friends_mask
        self.S = self.get_user_test(self.user_testRatings, self.user_testNegatives, self.friends, self.friends_mask)
        print("# Eval User-Item Interaction:", len(self.S))
        print("Finish loading U-I evaluating dataset.")

    def get_user_test(self, testRatings, testNegatives, friends, friends_mask):
        S = []
        for idx in range(len(testRatings)):
            rating = testRatings[idx]
            items = [rating[1]]
            items.extend(testNegatives[idx])
            user = np.full(len(items), rating[0])
            S.append([user, items, np.tile(friends[rating[0]], (len(items), 1)), np.tile(friends_mask[rating[0]], (len(items), 1))])
        return S

    def __len__(self):
        return len(self.S)

    def __getitem__(self, index):
        user, item, friends, friends_mask = self.S[index]
        user = torch.from_numpy(np.array(user, dtype=np.int64))
        item = torch.from_numpy(np.array(item, dtype=np.int64))
        friends = torch.from_numpy(np.array(friends, dtype=np.int64))
        friends_mask = torch.from_numpy(np.array(friends_mask, dtype=np.int64))
        return user, item, friends, friends_mask



class GroupItemDataset(data.Dataset):
    def __init__(self, path, num_negatives, num_users, num_items, device):
        # group data
        self.device = device
        self.path = path
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives  # 10
        self.group_trainMatrix = load_rating_file_as_matrix(path + "groupRatingTrain.txt")
        self.num_groups, self.num_group_net_items = self.group_trainMatrix.shape

        self.max_gsize = 0
        self.adj, group_data, self.group_member_dict = self.get_hyper_adj(path + 'groupMember.txt', path + "groupRatingTrain.txt")
        self.members, self.members_mask = self.get_member_mask()
        self.D, self.A = self.get_group_adj(group_data)
        self.S = None

        print("# Train Group-Item Interaction:", self.group_trainMatrix.nnz * self.num_negatives)
        print("Finish loading G-I training dataset.")

    def __len__(self):
        return self.group_trainMatrix.nnz * self.num_negatives

    def __getitem__(self, index):
        group, pos, neg, members, members_mask = self.S[index]
        group = torch.from_numpy(np.array(group, dtype=np.int64))
        pos = torch.from_numpy(np.array(pos, dtype=np.int64))
        neg = torch.from_numpy(np.array(neg, dtype=np.int64))
        members = torch.from_numpy(np.array(members, dtype=np.int64))
        members_mask = torch.from_numpy(np.array(members_mask, dtype=np.int64))
        return group, pos, neg, members, members_mask

    def get_member_mask(self):
        m_m_d = {}
        m_mask_d = {}
        for i in range(self.num_groups):
            group = i
            members = self.group_member_dict[i]
            cur_len = len(members)
            members_masked = np.append(members, np.zeros(self.max_gsize - cur_len))
            mask = np.zeros(self.max_gsize)
            mask[cur_len:] = 1.0
            m_m_d[group] = members_masked
            m_mask_d[group] = mask
        return m_m_d, m_mask_d

    def get_hyper_adj(self, user_in_group_path, group_train_path):
        g_m_d = {}
        max_gsize = 0
        with open(user_in_group_path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                max_gsize = max(max_gsize, len(g_m_d[g]))
                line = f.readline().strip()
        self.max_gsize = max_gsize

        g_i_d = defaultdict(list)
        with open(group_train_path, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    group, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        g_i_d[group].append(item + self.num_users)
                else:
                    group, item = int(arr[0]), int(arr[1])
                    g_i_d[group].append(item + self.num_users)
                line = f.readline()

        group_data = []
        for i in range(self.num_groups):
            group_data.append(g_m_d[i] + g_i_d[i])

        def _data_masks(all_group_data):
            indptr, indices, data = [], [], []
            indptr.append(0)
            for j in range(len(all_group_data)):
                single_group = np.unique(np.array(all_group_data[j]))
                length = len(single_group)
                s = indptr[-1]
                indptr.append(s + length)
                for i in range(length):
                    indices.append(single_group[i])
                    data.append(1)
            matrix = sp.csr_matrix((data, indices, indptr), shape=(self.num_groups, self.num_users + self.num_items))
            return matrix

        H_T = _data_masks(group_data)
        BH_T = H_T.T.multiply(1.0/(1.0 + H_T.sum(axis=1).reshape(1, -1)))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0/(1.0 + H.sum(axis=1).reshape(1, -1)))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)
        return DHBH_T.tocoo(), group_data, g_m_d


    def get_group_adj(self, group_data):
        try:
            matrix = load(self.path + "group_adj.dill")
        except:
            matrix = np.zeros((self.num_groups, self.num_groups))
            for i in tqdm(range(self.num_groups)):
                group_a = set(group_data[i])
                for j in range(i + 1, self.num_groups):
                    group_b = set(group_data[j])
                    overlap = group_a.intersection(group_b)
                    ab_set = group_a | group_b
                    matrix[i][j] = float(len(overlap) / len(ab_set))
                    matrix[j][i] = matrix[i][j]
            matrix = matrix + np.diag([1.0] * self.num_groups)
            save(matrix, self.path + "group_adj.dill")
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    def get_group_train_instances(self, train, members, members_mask):
        num_users = train.shape[0]
        num_items = train.shape[1]
        S = []
        for (u, i) in train.keys():
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                S.append([u, i, j, members[u], members_mask[u]])
        return S


class GroupEvalDataset(data.Dataset):
    def __init__(self, path, members, members_mask, num_items):
        self.group_testRatings = load_rating_file_as_list(path + "groupRatingTest.txt")
        self.group_testNegatives = load_negative_file(path + "groupRatingNegative.txt", num_items)
        self.members, self.members_mask = members, members_mask
        self.S = self.get_group_test(self.group_testRatings, self.group_testNegatives, self.members, self.members_mask)
        print("# Eval Group-Item Interaction:", len(self.S))
        print("Finish loading G-I evaluating dataset.")

    def get_group_test(self, testRatings, testNegatives, members, members_mask):
        S = []
        for idx in range(len(testRatings)):
            rating = testRatings[idx]
            items = [rating[1]]
            items.extend(testNegatives[idx])
            group = np.full(len(items), rating[0])
            S.append([group, items, np.tile(members[rating[0]], (len(items), 1)), np.tile(members_mask[rating[0]], (len(items), 1))])
        return S

    def __len__(self):
        return len(self.S)

    def __getitem__(self, index):
        group, item, members, merbers_mask = self.S[index]
        group = torch.from_numpy(np.array(group, dtype=np.int64))
        item = torch.from_numpy(np.array(item, dtype=np.int64))
        members = torch.from_numpy(np.array(members, dtype=np.int64))
        merbers_mask = torch.from_numpy(np.array(merbers_mask, dtype=np.int64))
        return group, item, members, merbers_mask

