import torch
import torch.nn.functional as F
from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/S3DIS/input',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='data/S3DIS/initial_superpoints',
                        help='initial superpoint path')
    parser.add_argument('--save_path', type=str, default='trained_models/S3DIS/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=12, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()


def eval_once(args, model, test_loader, classifier, use_sp=False):

    all_preds, all_label = [], []
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            region = region.squeeze()
            #
            if use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats[valid_mask].mean(0, keepdim=True))
                region_feats = torch.cat(region_feats, dim=0)
                #
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

                region_scores = F.linear(F.normalize(region_feats), F.normalize(classifier.weight))
                region_no = 0
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        preds[valid_mask] = torch.argmax(region_scores, dim=1).cpu()[region_no]
                        region_no +=1
            else:
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map.long()]
            all_preds.append(preds[labels!=args.ignore_label]), all_label.append(labels[[labels!=args.ignore_label]])

    return all_preds, all_label



def eval(epoch, args, test_areas = ['Area_5']):
    # First model and cls variables are set. The model variable holds the feature extraction model (I think) and the cls variable stores s-centroids classifier
    # used during training by clustering all superpoints across all point clouds into S semantic primatives (300) to create pseudo labels for training the feature 
    # extraction model.


    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    # initializing an empty/generic model with correct input and output dimensions 
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    # loading the model weights from the save path: 'trained_models/S3DIS/' into the empty model. Checkpoints.pth are stored by pytorch during training,
    # they save the model weights after each epoch during training. In this case, as seen in the main function below, the epoch 1270 was used for evaluation. 
    # I am guessing this epoch had the best training and testing accuracy, meaning the best trade-off between over- and underfitting. 
    model.eval()
    # This puts the model into evaluation mode, meaning the weights are not changed and I think the gradients are not stored. Something like that, its not
    # really relavant here

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()
    # Here the same: An empty linear classifier is initialized, the weights are loaded into it from the _checkpoint.pth file created during training and
    # the mode is set into evalmode. 


    primitive_centers = cls.weight.data###[300, 128]
    print('Merging Primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0, n_jobs=10).fit_predict(primitive_centers.cpu().numpy())#.astype(np.float64))
    # The cls weights are the centroids of the 300 S sematic primatives. These are clustered again with K-means into n_clusters, the final desired number of clusters.
    # n_init only means, that the algorithm is run 10 times with different initializations and the best clustering is selected as the Otput of this clustering,
    # random_state is like the seed.
    
    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # computes the centroids of the C clusteres. This means, that for each cluster, the semantic primatives, that were assigned to that cluster are averaged.
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    # This function, get_fixclassifier, which the GrowSP people wrote in lib/utils.py only creates a simple K-means based classifier. 
    # This means the classifier variable can be used to take an input and assign a cluster to it based on the proximity to a cluster center.
    classifier.eval()

    test_dataset = S3DIStest(args, areas=test_areas)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=4, pin_memory=True)
    # This is Pytorch creating a Dataset instance and a Dataloader insatance with the test dataset. The Dataloader depends on the cfl_collate_function, 
    # which created batches from single datapoints returned by the __getitem__ function of the Dataset instance.
    # The most important thing to note here 
    # is the fact, that the data loader loads from the .ply files created at the very first step of the pipeline and does not take the initialSP .npy 
    # files into account. The Dataloader returns all the information for a single datapoint with the __getitem__ function, the information on the superpoints 
    # are returned in the region variable, which is not used here since the use_sp variable is set to false in the eval_once function, turning of the if-section 
    # where the region variable is used.

    preds, labels = eval_once(args, model, test_loader, classifier)
    # eval_once (defined aboe) creates 
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum()*100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1))*100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]

    '''Final Metrics'''
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)

    return o_Acc, m_Acc, s


if __name__ == '__main__':

    args = parse_args()
    for epoch in range(10, 1500):
        if epoch % 1270 == 0:
            o_Acc, m_Acc, s = eval(epoch, args)
            print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)
