import torch
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from config import Config
from Dataset import GroupItemDataset, UserItemDataset, GroupEvalDataset, UserEvalDataset
from model import DisRec
from time import time
from util import getHitK, getNdcgK
from tqdm import tqdm


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    config = Config()

    set_seed(43)

    device_id = "cuda"
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    user_dataset = UserItemDataset(config.path, config.num_negatives, config.social_num, device)
    user_eval_dataset = UserEvalDataset(config.path, user_dataset.friends, user_dataset.friends_mask, 100)
    group_dataset = GroupItemDataset(config.path, config.num_negatives, user_dataset.num_users, user_dataset.num_items, device)
    group_eval_dataset = GroupEvalDataset(config.path, group_dataset.members, group_dataset.members_mask, user_dataset.num_items)

    ui_train_loader = DataLoader(user_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.work_number,
                                 pin_memory=True)
    ui_eval_loader = DataLoader(user_eval_dataset, batch_size=config.eval_size, shuffle=True,
                                num_workers=config.work_number,
                                pin_memory=True)
    gi_train_loader = DataLoader(group_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.work_number,
                                 pin_memory=True)
    gi_eval_loader = DataLoader(group_eval_dataset, batch_size=config.eval_size, shuffle=True, num_workers=config.work_number,
                                 pin_memory=True)


    model = DisRec(config, group_dataset, user_dataset, device)
    model = model.to(device)

    #  training user embedding
    print("Start training user embeddings.")
    for epoch in range(config.user_epoch):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        user_dataset.S = user_dataset.get_user_train_instances(user_dataset.user_trainMatrix, user_dataset.friends,
                                                               user_dataset.friends_mask)
        model.train()
        train_user_loss = 0.0
        start_time = time()
        for batch_index, data in tqdm(enumerate(ui_train_loader)):
            data = [x.to(device, non_blocking=True) for x in data]
            (user, pos_item, neg_item, friends, friends_mask) = data
            pos_predict = model(None, user, pos_item, friends, friends_mask, None, None)
            neg_predict = model(None, user, neg_item, friends, friends_mask, None, None)

            model.zero_grad()
            social_loss = torch.mean((pos_predict - neg_predict - 1) ** 2)
            preference_loss, reg_loss = model.preference.bpr_loss(user, pos_item, neg_item)
            reg_loss = reg_loss * config.decay
            preference_loss = preference_loss + reg_loss
            loss = social_loss + preference_loss
            loss.backward()
            optimizer.step()
            train_user_loss += loss

        elapsed = time() - start_time
        print('| epoch {:3d} |  time {:4.2f} | User loss {:4.4f}'.format(epoch, elapsed,
                                                                         train_user_loss / len(ui_train_loader)))
        if epoch % config.visual == 0:
            evaluate_time = time()
            hits, ndcgs = [], []
            pred_rank = None
            model.eval()
            with torch.no_grad():
                for batch_idx, data in tqdm(enumerate(ui_eval_loader)):
                    data = [x.to(device, non_blocking=True) for x in data]
                    (user, item, friends, friends_mask) = data
                    bsz = user.shape[0]
                    item_len = user.shape[1]
                    user = user.view(-1)
                    item = item.view(-1)
                    friends_len = friends.shape[2]
                    friends = friends.reshape(-1, friends_len)
                    friends_mask = friends_mask.reshape(-1, friends_len)
                    predictions = model(None, user, item, friends, friends_mask, None, None)
                    predictions = torch.reshape(predictions, (bsz, item_len))
                    pred_score = predictions.data.cpu().numpy()
                    temp_pred_rank = np.argsort(pred_score * -1, axis=1)
                    if pred_rank is None:
                        pred_rank = temp_pred_rank
                    else:
                        pred_rank = np.concatenate((pred_rank, temp_pred_rank), axis=0)
                for k in config.topK:
                    hits.append(getHitK(pred_rank, k))
                    ndcgs.append(getNdcgK(pred_rank, k))
            print(
                'User Epoch %d [%.1f s]: \n HR@5 = %.4f, HR@10 = %.4f; '
                'NDCG@5 = %.4f, NDCG@10 = %.4f [%.1f s]' % (
                epoch, elapsed, hits[0], hits[1], ndcgs[0], ndcgs[1], time() - evaluate_time))


    print("Start training group embeddings.")
    for epoch in range(config.group_epoch):
        optimizer_g = optim.Adam(model.parameters(), config.lr)
        group_dataset.S = group_dataset.get_group_train_instances(group_dataset.group_trainMatrix,
                                                                  group_dataset.members, group_dataset.members_mask)
        model.train()
        train_group_loss = 0.0
        start_time = time()
        for batch_index, data in tqdm(enumerate(gi_train_loader)):
            data = [x.to(device, non_blocking=True) for x in data]
            (group, pos_item, neg_item, members, members_mask) = data
            pos_predict, y2 = model(group, None, pos_item, None, None, members, members_mask)
            neg_predict, _ = model(group, None, neg_item, None, None, members, members_mask)
            model.zero_grad()
            loss = torch.mean((pos_predict - neg_predict - 1) ** 2)
            contra_loss = torch.mean(y2)
            loss = loss + config.delta * contra_loss

            loss.backward()
            optimizer_g.step()
            train_group_loss += loss

        elapsed = time() - start_time
        print('| epoch {:3d} |  time {:4.2f} | Group loss {:4.4f}'.format(epoch, elapsed,
                                                                          train_group_loss / len(gi_train_loader)))

        if epoch % config.visual == 0:
            evaluate_time = time()
            hits, ndcgs = [], []
            pred_rank = None
            model.eval()
            with torch.no_grad():
                for batch_idx, data in tqdm(enumerate(gi_eval_loader)):
                    data = [x.to(device, non_blocking=True) for x in data]
                    (group, item, members, members_mask) = data
                    bsz = group.shape[0]
                    item_len = group.shape[1]
                    group = group.view(-1)
                    item = item.view(-1)
                    group_len = members.shape[2]
                    members = members.reshape(-1, group_len)
                    members_mask = members_mask.reshape(-1, group_len)
                    predictions = model(group, None, item, None, None, members, members_mask)
                    predictions = torch.reshape(predictions, (bsz, item_len))
                    pred_score = predictions.data.cpu().numpy()
                    temp_pred_rank = np.argsort(pred_score * -1, axis=1)
                    if pred_rank is None:
                        pred_rank = temp_pred_rank
                    else:
                        pred_rank = np.concatenate((pred_rank, temp_pred_rank), axis=0)
                for k in config.topK:
                    hits.append(getHitK(pred_rank, k))
                    ndcgs.append(getNdcgK(pred_rank, k))
            print(
                'Group Epoch %d [%.1f s]: \n HR@5 = %.4f, HR@10 = %.4f; '
                'NDCG@5 = %.4f, NDCG@10 = %.4f [%.1f s]' % (
                epoch, elapsed, hits[0], hits[1], ndcgs[0], ndcgs[1], time() - evaluate_time))

    print("Done!")
