class Config(object):
    def __init__(self):
        self.dataset = 'Mafengwo'
        self.path = './data/' + self.dataset + '/'
        self.u_emb_size = 64
        self.g_emb_size = 2 * self.u_emb_size
        self.group_epoch = 100
        self.user_epoch = 50
        self.num_negatives = 10
        self.social_num = 20
        self.layers = 3
        self.batch_size = 512
        self.eval_size = 4
        self.lr = 0.001
        self.drop_ratio = 0.1
        self.topK = [5, 10]
        self.balance = 6
        self.gpu_id = 0
        self.decay = 1e-4
        self.delta = 0.7
        self.work_number = 0
        self.visual = 1
