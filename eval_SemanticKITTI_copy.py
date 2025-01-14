import torch
import torch.nn.functional as F
from datasets.ConstSite import ConstSitetest, cfl_collate_fn_test

import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier
from lib.helper_ply import read_ply, write_ply
import warnings
import argparse
import random
import os
import matplotlib.pyplot as plt

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    # parser.add_argument('--data_path', type=str, default='/workspace/data/S3DIS/input',
    #                     help='pont cloud data path')
    parser.add_argument('--data_path', type=str, default='/workspace/test_data',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='',
                        help='initial superpoint path')
    parser.add_argument('--save_path', type=str, default='trained_models/SemanticKITTI/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=10, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.15, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=3, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=500, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=19, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()

colormap = []
for _ in range(1000):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)

def eval_once(args, model, test_loader, classifier,use_sp=False):

    all_preds, all_label = [], []
    print('in eval_once')
    for data in test_loader:
        print('looping over data')
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data
            index = index[0]
            print('data index?')
            print(index)
            # print(dir(test_loader.dataset))
            file = test_loader.dataset.file[index]
            name = test_loader.dataset.name[index]
            print('file_name: ', file, ' ', name)
            print('coords in eval_once')
            print(coords.shape)
            print(features.shape)
            print(labels.shape)
            print(inverse_map.shape)
            # print(coords[:3,:])
            # print(np.unique(coords[:,0]))
            # print(np.unique(coords[:,1]))
            # print(np.unique(coords[:,2]))
            # print(np.unique(coords[:,3]))
            data = read_ply(file)
            # print(features.shape)

            # in_field = ME.TensorField(features, coords, device=0)
            in_field = ME.TensorField(coords[:, 1:] * args.voxel_size, coords, device=0)
            print('in_field')
            print(type(in_field))
            print(in_field.shape)
            feats = model(in_field)
            print('feats')
            print(feats.shape)
            feats = F.normalize(feats, dim=1)
            print(feats.shape)

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
                # print('in else')
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()
                print('preds after ARGMAX')
                print(preds.shape)
                print('scores')
                print(scores.shape)

            # print('inv map')
            # print(inverse_map.shape)
            preds = preds[inverse_map.long()]
            labels = labels[inverse_map.long()]
            vis_path = '/workspace/data/test_data/SeemanticKITTI_GrowSP/'
            coords =  np.vstack((data['x'], data['y'], data['z'])).T
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            colors = np.zeros_like(coords)
            for p in range(preds.shape[0]):
                # print(p)
                # print(preds[p])
                # print(preds[p].item())
                colors[p] = 255 * (colormap[preds[p].item()])[:3]
            colors = colors.astype(np.uint8)
 
            
            write_ply(vis_path + name, [coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
            print('preds again')
            print(preds.shape)
            print(preds)
            print('labels')
            print(labels.shape)
            print(np.unique(preds))
            print(np.unique(labels))
            print(args.ignore_label)
            all_preds.append(preds[labels!=args.ignore_label]), all_label.append(labels[[labels!=args.ignore_label]])
            print('all_preds')
            print(len(all_preds))
            print(all_preds)

    return all_preds, all_label



def eval(epoch, args):

    # model trained in training phase used in eval_once() to extract point features for each point cloud and cluster them
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()
    # .eval() just sets the mode od the model for evaluation not storing gradients and stuff like batch normalization

    # cls stores the features of the semantic primitives (roughly 300-500) which were derived from clustering all the points in the training set (i think across all training data)
    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()

    #the primatives are clustered (agian) into the desired number of semantic classes
    primitive_centers = cls.weight.data###[500, 128]
    print('Merging Primitives')
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=5, random_state=0, n_jobs=5).fit_predict(primitive_centers.cpu().numpy())#.astype(np.float64))

    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    
    #the centroids of the desired number of classes (19 for S3Dis) are used to create a simple kmeans classifier 
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    val_dataset = ConstSitetest(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=args.workers, pin_memory=True)

    # preds, labels = eval_once(args, model, val_loader, classifier)
    preds_res = eval_once(args, model, val_loader, classifier)
    print("CONCATENATING PREDS")
    print(type(preds_res))
    print(len(preds_res))
    print(type(preds_res[0]), type(preds_res[1]))
    print(type(preds_res[0][0]), type(preds_res[1][0]))
    preds = preds_res
    all_preds = torch.cat(preds_res).numpy()
    # all_labels = torch.cat(labels).numpy()

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
    # for epoch in range(1, 500):
    #     if epoch%400==0:
    #         o_Acc, m_Acc, s = eval(epoch, args)
    #         print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)
    all_preds = eval(400, args)
    print('all_preds collected: ')
    print(all_preds)

