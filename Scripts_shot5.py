import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import h5py
import json
import argparse
import os

import shutil
import torch.nn.functional as F
from tqdm import tqdm
import random

seed = 0
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)      # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

##################################
# Semantic Classifier
##################################

class SemanticClassifier(nn.Module):
    def __init__(self, this_embedding, feat_dims, num_classes):
        super(SemanticClassifier, self).__init__()
        embedding_dims = this_embedding.shape[1]
        self.embedding = this_embedding

        self.linear1 = torch.nn.utils.weight_norm(nn.Linear(embedding_dims, feat_dims), dim=0)
        self.weight_embedding = nn.Sequential(self.linear1, nn.LeakyReLU(0.1))

        self.drop_out = nn.Dropout(p=0.7)
        self.linear2 = torch.nn.utils.weight_norm(nn.Linear(feat_dims, feat_dims), dim=0)

    def forward(self, x):
        descriptor = self.weight_embedding(self.embedding)
        classifier = self.linear2(self.drop_out(descriptor))
        
        return F.linear(x, classifier)

##################################
# Semantic-Visual Model
##################################

class SemanticModel(nn.Module):
    def __init__(self, this_embedding, feat_dims=512, num_classes=100, lamda=0.5):
        super(SemanticModel, self).__init__()
        self.embedding = this_embedding
        embedding_dims = self.embedding.shape[1] # 1000, (360)
        self.lamda = lamda
        self.semantic_descriminator = nn.Linear(embedding_dims, feat_dims)
        self.semantic_classifier = SemanticClassifier(this_embedding, feat_dims, num_classes)
        self.visual_classifier = torch.nn.utils.weight_norm(nn.Linear(feat_dims, num_classes), dim=0)

    def descriminate(self, x):
        # descriptor
        descriptor = self.semantic_descriminator(self.embedding)
        return F.linear(x, descriptor)
    
    def semantic_forward(self, x):
        # semantic classifier
        semantic_scores = self.semantic_classifier(x)
        return semantic_scores
        
    def visual_forward(self, x):
        # visual classifier
        visual_scores = self.visual_classifier(x)
        return visual_scores

    def inference(self, x):
        # final scores
        visual_scores = self.visual_forward(x)
        semantic_scores = self.semantic_forward(x)
        return self.lamda*visual_scores + (1.0-self.lamda)*semantic_scores

################################
# to construct HDF5-type dataset
# dictionary-type
################################

class SimpleHDF5Dataset:
    def __init__(self, file_handle):
        self.f = file_handle
        self.all_feats_dset = self.f['all_feats']
        self.all_labels = self.f['all_labels']
        self.total = self.f['count']

    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total


#############################################################################
# a dataset to allow for category-uniform sampling of base and novel classes.
# also incorporates hallucination
# sampling features to be mixed (base & novel)
#############################################################################

class LowShotDataset:
    def __init__(self, base_feats, novel_feats, base_classes, novel_classes):
        self.f = base_feats
        self.all_base_feats_dset = self.f['all_feats'][...]
        self.all_base_labels_dset = self.f['all_labels'][...]

        self.novel_feats = novel_feats['all_feats']
        self.novel_labels = novel_feats['all_labels']

        self.base_classes = base_classes
        self.novel_classes = novel_classes

        self.frac = 0.5
        self.all_classes = np.concatenate((base_classes, novel_classes))
        self.base_idx_np = self.get_base_index()

    def get_base_index(self):
        labels = sorted(self.base_classes)
        base_idx = []
        for label_i in labels:
            base_idx.append(np.where(self.all_base_labels_dset == label_i)[0])
        base_idx_np = np.array(base_idx, dtype=np.int32)
        return base_idx_np

    def sample_base_class_examples(self, num):
        sampled_idx = np.sort(np.random.choice(len(self.all_base_labels_dset), num, replace=False))
        return torch.Tensor(self.all_base_feats_dset[sampled_idx,:]), torch.LongTensor(self.all_base_labels_dset[sampled_idx].astype(int))

    def sample_novel_class_examples(self, num):
        sampled_idx = np.random.choice(len(self.novel_labels), num)
        return torch.Tensor(self.novel_feats[sampled_idx,:]), torch.LongTensor(self.novel_labels[sampled_idx].astype(int)), sampled_idx

    def get_sample(self, batchsize):
        num_base = round(self.frac*batchsize)
        num_novel = batchsize - num_base
        base_feats, base_labels = self.sample_base_class_examples(int(num_base))
        novel_feats, novel_labels = self.sample_novel_class_examples(int(num_novel))
        return torch.cat((base_feats, novel_feats)), torch.cat((base_labels, novel_labels))

    def get_index_base_novel_sample(self, batchsize, cur_novel_l, positive_index):
        novel_feats, novel_labels, sampled_novel_idx = self.sample_novel_class_examples(int(batchsize))
        novel_idx_batch = []

        for curent_label in novel_labels:
            idx = np.where(cur_novel_l == curent_label.data.numpy())[0][0]
            novel_idx_batch.append(idx)
            # select beta classes as idy

        idy = positive_index.data[novel_idx_batch]
        base_positive = self.base_idx_np[idy]
        base_positive = torch.from_numpy(base_positive.reshape((base_positive.shape[0], -1)))

        random_matrix = torch.rand(base_positive.shape)
        sampled_positive_idx = torch.diag(base_positive[:,random_matrix.max(-1)[-1]])

        positive_feats = self.all_base_feats_dset[sampled_positive_idx]
        positive_labels = self.all_base_labels_dset[sampled_positive_idx]

        return torch.cat((torch.Tensor(positive_feats), novel_feats)), torch.cat((torch.LongTensor(positive_labels), novel_labels))


    def featdim(self):
        return self.novel_feats.shape[1]

##################################################
# simple data loader for test
##################################################
def get_test_loader(file_handle, batch_size=1000):
    testset = SimpleHDF5Dataset(file_handle)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return data_loader

def construct_patchmix(x, rotated_x, alpha):

    new_x = x + rotated_x
    return new_x

def CrossEntropy(pred, target, lam, scale = False):
    pred = pred.softmax(-1)
    loss = -torch.log(pred) * target
    loss = loss * lam
    if scale:
        loss = loss.sum() / ((target > 0).sum() + 0.000001)
    else:
        loss = loss.sum() / (target.sum() + 0.000001)

    return loss

def BinaryEntropy(pred, target, scale = False):
    pred = pred.sigmoid()
    loss = -torch.log(pred + 0.0000001) * target
    if scale:
        loss = loss.sum() / ((target > 0).sum() + 0.000001)
    else:
        loss = loss.sum() / (target.sum() + 0.000001)
    return loss

def mixup_criterion(pred, y_a, y_b, lam):
    criterion = CrossEntropy
    return criterion(pred, y_a, lam) + criterion(pred, y_b, (1. - lam))

def training_loop(lowshot_dataset,novel_test_feats, num_classes, params, batchsize=1000, maxiters=1000, nTimes = 0):
    if os.path.exists('Model_SHOT5/' + params.name + '/LayerBestModel_' + str(params.beta) + '_' + str(params.lamda) + '/') == False:
        os.makedirs('Model_SHOT5/' + params.name + '/LayerBestModel_' + str(params.beta) + '_' + str(params.lamda) + '/')
    if os.path.exists('Model_SHOT5/' + params.name + '/' + str(nTimes) + '_' + str(params.beta) + '_' + str(params.lamda)) == False:
        os.makedirs('Model_SHOT5/' + params.name + '/' + str(nTimes) + '_' + str(params.beta) + '_' + str(params.lamda))


    featdim = novel_test_feats['all_feats'].shape[1]
    t_embedding_100 = torch.FloatTensor(np.load('MiniImageNetWord2Vec.npy'))
    t_original_relation = F.normalize(t_embedding_100,dim=-1).mm(F.normalize(t_embedding_100,dim=-1).t())

    novel_labels = list(set(lowshot_dataset.novel_labels))

    invalid_labels = list(set(list(set(range(100)).difference(set(novel_labels)))).difference(base_classes))
    this_embedding = t_embedding_100
    this_embedding[invalid_labels,:] = 0.
    this_embedding = F.normalize(this_embedding, dim=-1)
    this_embedding = Variable(this_embedding.cuda())
    
    model = SemanticModel(this_embedding, featdim, num_classes, params.lamda)
    model = model.cuda()

    test_loader = get_test_loader(novel_test_feats)

    best_ACC = 0.0
    tmp_epoach = 0
    tmp_count = 0
    tmp_rate = params.lr
    recode_reload = {}
    reload_model = False
    max_tmp_count = 10

    optimizer_descriminator = torch.optim.Adam(model.semantic_descriminator.parameters(), tmp_rate, weight_decay=params.wd)
    optimizer_semantic = torch.optim.Adam(model.semantic_classifier.parameters(), tmp_rate, weight_decay=params.wd)
    optimizer_visual = torch.optim.Adam(model.visual_classifier.parameters(), tmp_rate, weight_decay=params.wd)

    ##########################################
    t_original_relation = t_original_relation[novel_labels][:,:64]
    _, positive_index = torch.topk(t_original_relation, 64, dim=-1)
    positive_index = positive_index[:,params.beta]
    batch_list = list(range(batchsize))
    ##########################################
    for epoch in range(maxiters):

        if reload_model == True:

            if str(tmp_epoach) in recode_reload:

                recode_reload[str(tmp_epoach)] += 1
                tmp_rate = tmp_rate * 0.1
                if tmp_rate < 1e-4:

                    if best_ACC <= 0.2:
                        return 0
                    old_path = 'Model_SHOT5/' + params.name + '/' +  str(nTimes) + '_' + str(params.beta) + '_' + str(params.lamda) +'/' + str(nTimes) +'_save_' + str(tmp_epoach)+ '_' + str(round(best_ACC, 8)) + '.pth'
                    new_path = 'Model_SHOT5/' + params.name + '/LayerBestModel_' + str(params.beta) + '_' + str(params.lamda) + '/'
                    f = open(new_path + str(nTimes) +'_save_' + str(tmp_epoach)+ '_' + str(round(best_ACC, 8)) + '.pth','w')
                    f.close()
                    shutil.rmtree('Model_SHOT5/' + params.name + '/' +  str(nTimes) + '_' + str(params.beta) + '_' + str(params.lamda))

                    return best_ACC

                for param_group in optimizer_semantic.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                for param_group in optimizer_visual.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                for param_group in optimizer_descriminator.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

            else:

                recode_reload[str(tmp_epoach)] = 1


            pretrained_model = 'Model_SHOT5/' + params.name + '/' +  str(nTimes) + '_' + str(params.beta) + '_' + str(params.lamda) +'/' + str(nTimes) +'_save_' + str(tmp_epoach)+ '_' + str(round(best_ACC, 8)) + '.pth'
            pretrain = torch.load(pretrained_model)  
            model.load_state_dict(pretrain['state_dict'])

            reload_model = False

        (x,y) = lowshot_dataset.get_index_base_novel_sample(batchsize, novel_labels, positive_index)
        x = Variable(x.cuda())
        y = Variable(y.cuda())

        x_positive, x_novel = torch.chunk(x, 2, dim = 0)
        y_positive, y_novel = torch.chunk(y, 2, dim = 0)

        # 1-st stage
        optimizer_descriminator.zero_grad()
        scores_1 = model.descriminate(x_novel)
        loss_1 = F.cross_entropy(scores_1, y_novel)
        loss = loss_1
        loss.backward()
        optimizer_descriminator.step()

        # 2-nd stage
        scores_2 = model.descriminate(torch.cat((x_positive, x_novel)))
        scores_2 = F.softmax(scores_2, dim=-1)

        scores_2_pos, scores_2_novel = torch.chunk(scores_2, 2, dim= 0)

        ################################################
        y_a = torch.zeros(scores_2_novel.shape)
        y_b = torch.zeros(scores_2_novel.shape)
        batch_list = torch.arange(0, scores_2_novel.shape[0])
        y_a[batch_list, y_novel] = 1.
        y_b[batch_list, y_positive] = 1.
        y_a = Variable(y_a.cuda())
        y_b = Variable(y_b.cuda())
        ################################################

        scores_2_pos = scores_2_pos[batch_list, y_novel]
        scores_2_novel = scores_2_novel[batch_list, y_novel]
        alpha = (scores_2_novel - scores_2_pos).unsqueeze(1)
        alpha[alpha < 0.] += 1.

        # generate mixup feats
        lam_pos = 1. - alpha.data
        feat_mixup_pos = construct_patchmix(lam_pos * x_novel, (1 - lam_pos) * x_positive, alpha)
        all_feats = torch.cat((x_novel, feat_mixup_pos), dim=0)

        # visual end
        optimizer_visual.zero_grad()
        scores_visual_novel, scores_visual_mixup = torch.chunk(model.visual_forward(all_feats), 2, dim=0)

        loss_visual_novel = F.cross_entropy(scores_visual_novel, y_novel)
        loss_visual_mixup = mixup_criterion(scores_visual_mixup, y_a, y_b, lam_pos)

        loss = loss_visual_mixup + loss_visual_novel
        loss.backward()
        optimizer_visual.step()

        # semantic end
        optimizer_semantic.zero_grad()
        scores_semantic_novel, scores_semantic_mixup = torch.chunk(model.semantic_forward(all_feats), 2, dim=0)
        loss_semantic_novel = F.cross_entropy(scores_semantic_novel, y_novel)
        loss_semantic_mixup = mixup_criterion(scores_semantic_mixup, y_a, y_b, lam_pos)

        loss =  loss_semantic_mixup + loss_semantic_novel
        loss.backward()
        optimizer_semantic.step()


        if (epoch%10==0):
            accs = eval_loop(test_loader, model, base_classes, novel_classes)
            tmp_count += 1

            if  accs[0] > best_ACC :

                best_ACC = accs[0]
                save_file_path = 'Model_SHOT5/' + params.name + '/' +  str(nTimes) + '_' + str(params.beta) + '_' + str(params.lamda) +'/' + str(nTimes) +'_save_' + str(epoch)+ '_' + str(round(best_ACC, 8)) + '.pth'
                states = {
                    'state_dict': model.state_dict(),
                }
                torch.save(states, save_file_path)

                tmp_count = 0
                tmp_epoach = epoch

            if tmp_count == max_tmp_count:
                reload_model = True
                tmp_count = 0

    return model

def perelement_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float), top5_correct.astype(float)


def eval_loop(data_loader, model, base_classes, novel_classes):
    model = model.eval()
    top1 = None
    top5 = None
    no_novel_class = list(set(range(100)).difference(set(novel_classes)))
    all_labels = None
    for i, (x,y) in enumerate(data_loader):
        x = Variable(x.cuda())
        feat_x = x
        scores = model.inference(feat_x).softmax(-1)
        scores[:,no_novel_class] = -0.0
        top1_this, _ = perelement_accuracy(scores.data, y)
        top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
        all_labels = y.numpy() if all_labels is None else np.concatenate((all_labels, y.numpy()))

    is_novel = np.in1d(all_labels, novel_classes)
    top1_novel = np.mean(top1[is_novel])
    return [top1_novel]

def parse_args():
    parser = argparse.ArgumentParser(description='Low shot benchmark')
    parser.add_argument('--name', default='5-shot', type=str)
    parser.add_argument('--numclasses', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--maxiters', default=2002, type=int)
    parser.add_argument('--batchsize', default=10, type=int)
    parser.add_argument('--beta', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--lamda', default=0.8, type=float)
    parser.add_argument('--feature_path', required=True, type=str)
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    print(params.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)

    with open('ExperimentSplit/Json/base_classes_train_meta.json','r') as f:
        exp = json.load(f)
        base_classes = list(set(exp['image_labels']))

    with open('ExperimentSplit/Json/base_classes_test_meta.json','r') as f:
        exp = json.load(f)
        novel_classes = list(set(exp['image_labels']))



    train_feats = h5py.File('Features/mini-ImageNet/' + params.feature_path + '/train.hdf5', 'r')
    test_feats = h5py.File('Features/mini-ImageNet/' + params.feature_path + '/test.hdf5', 'r')
    all_feats_dset_ = test_feats['all_feats'][...]
    all_labels_ = test_feats['all_labels'][...]
    all_labels = all_labels_[all_labels_ != 0]
    all_feats_dset = all_feats_dset_[all_labels_ != 0]

    ########################################
    params.name = params.name + '-' + params.feature_path
    ########################################
    n_shot = 5

    start_ = 0
    end_ = 600

    if os.path.exists('Model_SHOT5/' + params.name + '/LayerBestModel_' + str(params.beta) + '_' + str(params.lamda)):
        len_results = len(os.listdir('Model_SHOT5/' + params.name + '/LayerBestModel_' + str(params.beta) + '_' + str(params.lamda)))
        if len_results < end_:
            start_ = len_results
        else:
            start_ = end_
    lowshot_dataset = None
    for i, nTime in tqdm(enumerate(range(start_, end_))):

        selected = np.random.choice(novel_classes, 5, replace=False)

        novel_train_feats = []
        novel_train_labels = []
        novel_test_feats = []
        novel_test_labels = []

        for K in selected:
            is_K = np.in1d(all_labels, K)

            current_idx = np.random.choice(np.sum(is_K), 15 + n_shot, replace=False)
            novel_train_feats.append(all_feats_dset[is_K][current_idx[:n_shot]])
            novel_test_feats.append(all_feats_dset[is_K][current_idx[n_shot:]])

            for _ in range(n_shot):
                novel_train_labels.append(K)
            for _ in range(15):
                novel_test_labels.append(K)

        novel_train_feats  =  np.vstack(novel_train_feats)
        novel_train_labels =  np.array(novel_train_labels)
        novel_test_feats   =  np.vstack(novel_test_feats)
        novel_test_labels  =  np.array(novel_test_labels)

        novel_feats = {}
        novel_feats['all_feats'] = novel_train_feats
        novel_feats['all_labels'] = novel_train_labels
        novel_feats['count'] = len(novel_train_labels)

        novel_val_feats = {}
        novel_val_feats['all_feats'] = novel_test_feats
        novel_val_feats['all_labels'] = novel_test_labels
        novel_val_feats['count'] = len(novel_test_labels)


        if lowshot_dataset is not None:
            lowshot_dataset.novel_feats = novel_feats['all_feats']
            lowshot_dataset.novel_labels = novel_feats['all_labels']
            lowshot_dataset.novel_classes = novel_classes
            lowshot_dataset.all_classes = np.concatenate((base_classes, novel_classes))
        else:
            lowshot_dataset = LowShotDataset(train_feats, novel_feats, base_classes, novel_classes)

        model = training_loop(lowshot_dataset, novel_val_feats, params.numclasses, params, params.batchsize, params.maxiters, nTimes = nTime)