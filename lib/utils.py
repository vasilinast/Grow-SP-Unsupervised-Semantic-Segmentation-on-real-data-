import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.cluster import KMeans
import MinkowskiEngine as ME
import time


def get_sp_feature(args, loader, model, current_growsp):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} computing point feats ....")
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    # print('before model eval')
    model.eval()
    # print('after model eval')
    context = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data
            # print('region shape', region.shape)
            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(features, coords, device=0)
            # print("checking Minkowski is on GPU?")
            # print(in_field.device) 

            feats = model(in_field)
            # feats = F.normalize(feats, dim=-1)
            feats = feats[inds.long()]

            valid_mask = region != -1
            # print(valid_mask.shape)
            '''Compute avg rgb/xyz/norm for each Superpoints to help merging superpoints'''
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            labels = labels.to("cuda")
            labels = labels[valid_mask]
            region = region.to("cuda")
            region = region[valid_mask].long()
            # print(f"features device: {features.device}")
            # print(f"normals device: {normals.device}")
            # print(f"labels device: {labels.device}")
            # print(f"feats device: {feats.device}")
            # print(f"region device: {region.device}")

            ##
            pc_rgb = features[:, 0:3]
            pc_xyz = features[:, 3:] * args.voxel_size
            ##
            region_num = len(torch.unique(region))
            # print("region num")
            # print(region)
            # print(region_num)
            region_corr = torch.zeros(region.size(0), region_num)  # ?
            # print(f"region device: {region.device}")
            region_corr = region_corr.to("cuda")
            # print(f"region_corr device: {region_corr.device}")
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()  ##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            # print(f"per_region_num device: {per_region_num.device}")
            ###
            region_feats = F.linear(region_corr.t(), feats.t()) / per_region_num
            # print(f"region_feats device: {region_feats.device}")

            if current_growsp is not None:
                # print('current_growsp IS NOT None')
                region_rgb = F.linear(region_corr.t(), pc_rgb.t()) / per_region_num
                region_xyz = F.linear(region_corr.t(), pc_xyz.t()) / per_region_num
                region_norm = F.linear(region_corr.t(), normals.t()) / per_region_num

                rgb_w, xyz_w, norm_w = args.w_rgb, args.w_xyz, args.w_norm
                region_feats = F.normalize(region_feats, dim=-1)
                region_feats = torch.cat((region_feats, rgb_w * region_rgb, xyz_w * region_xyz, norm_w * region_norm), dim=-1)
                #
                if region_feats.size(0) < current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
            else:
                # print('current_growsp IS None')
                # print(f"region_feats device: {region_feats.device}")
                feats = region_feats
                # print(f"feats device: {feats.device}")

                sp_idx = torch.tensor(range(region_feats.size(0)), device=0)

            # print(f"sp_idx device: {sp_idx.device}")
            neural_region = sp_idx[region]
            # print(f"neural_region device: {neural_region.device}")

            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            # print(neural_region_num)
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num, device=0)

            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            # print(f"per_neural_region_num device: {per_neural_region_num.device}")

            '''Compute avg rgb/pfh for each Superpoints to help Primitives Learning'''
            final_rgb = F.linear(neural_region_corr.t(), pc_rgb.t()) / per_neural_region_num
            # print(f"final_rgb device: {final_rgb.device}")

            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)

            # print("total number of unique neural regions: ", torch.unique(neural_region).shape)
            for p in torch.unique(neural_region):

                if p != -1:
                    mask = p == neural_region
                    pfh.append(compute_hist(normals[mask].cpu()).unsqueeze(0).cuda())
            pfh = torch.cat(pfh, dim=0)
            # print(f"pfh device: {pfh.device}")
            # print("pfh shape: ", pfh.shape)

            # unique_regions = torch.unique(neural_region)
            # # print(f"unique_regions device: {unique_regions.device}")
            # masks = (neural_region.unsqueeze(0) == unique_regions.unsqueeze(1))
            # pfh2 = compute_hist_vectorized_julia(normals, masks)
            # # print(f"pfh2 device: {pfh2.device}")
            # pfh = pfh.cpu() if pfh.is_cuda else pfh
            # pfh2 = pfh2.cpu() if pfh2.is_cuda else pfh2

            # are_equal = torch.all(torch.eq(pfh, pfh2))
            # # print(f"pfh and pfh2 are exactly equal: {are_equal}")
            # max_diff = torch.max(torch.abs(pfh - pfh2))
            # # print(f"Maximum absolute difference: {max_diff}")
            # mse = torch.mean((pfh - pfh2) ** 2)
            # # print(f"Mean Squared Error: {mse}")
            # are_close = torch.allclose(pfh, pfh2, rtol=1e-5, atol=1e-8)
            # # print(f"pfh and pfh2 are close within tolerance: {are_close}")

            feats = F.normalize(feats, dim=-1)
            # print(f"feats device: {feats.device}")
            feats = torch.cat((feats, args.c_rgb * final_rgb, args.c_shape * pfh), dim=-1)
            # print(f"feats device: {feats.device}")
            feats = F.normalize(feats, dim=-1)
            # print(f"feats device: {feats.device}")

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sp_index.append(neural_region)
            context.append((scene_name, gt, raw_region))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}- returning from utils.get_sp_feature")
    return point_feats_list, point_labels_list, all_sp_index, context

def compute_hist_vectorized_julia(normals, masks, bins=10, min=-1, max=1):
    normals = F.normalize(normals, dim=-1)
    batch_size = masks.shape[0]
    hist_list = []
    
    for i in range(batch_size):
        mask = masks[i]
        normal = normals[mask]
        relation = torch.mm(normal, normal.t())
        relation = torch.triu(relation, diagonal=0)
        hist = torch.histc(relation, bins, min, max)
        hist /= hist.sum()
        hist_list.append(hist)
    
    return torch.stack(hist_list)

def get_kittisp_feature(args, loader, model, current_growsp):
    print('computing point feats ....')
    point_feats_list = []
    point_labels_list = []
    all_sp_index = []
    model.eval()
    context = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index = data

            region = region.squeeze()
            scene_name = loader.dataset.name[index[0]]
            gt = labels.clone()
            raw_region = region.clone()

            in_field = ME.TensorField(coords[:, 1:]*args.voxel_size, coords, device=0)

            feats = model(in_field)
            feats = feats[inds.long()]

            valid_mask = region!=-1
            features = features[inds.long()].cuda()
            features = features[valid_mask]
            normals = normals[inds.long()].cuda()
            normals = normals[valid_mask]
            feats = feats[valid_mask]
            labels = labels[valid_mask]
            region = region[valid_mask].long()
            ##
            pc_remission = features
            ##
            region_num = len(torch.unique(region))
            region_corr = torch.zeros(region.size(0), region_num)#?
            region_corr.scatter_(1, region.view(-1, 1), 1)
            region_corr = region_corr.cuda()##[N, M]
            per_region_num = region_corr.sum(0, keepdims=True).t()
            ###
            region_feats = F.linear(region_corr.t(), feats.t())/per_region_num
            if current_growsp is not None:
                region_feats = F.normalize(region_feats, dim=-1)
                #
                if region_feats.size(0) < current_growsp:
                    n_segments = region_feats.size(0)
                else:
                    n_segments = current_growsp
                sp_idx = torch.from_numpy(KMeans(n_clusters=n_segments, n_init=5, random_state=0, n_jobs=5).fit_predict(region_feats.cpu().numpy())).long()
            else:
                feats = region_feats
                sp_idx = torch.tensor(range(region_feats.size(0)))

            neural_region = sp_idx[region]
            pfh = []

            neural_region_num = len(torch.unique(neural_region))
            neural_region_corr = torch.zeros(neural_region.size(0), neural_region_num)
            neural_region_corr.scatter_(1, neural_region.view(-1, 1), 1)
            neural_region_corr = neural_region_corr.cuda()
            per_neural_region_num = neural_region_corr.sum(0, keepdims=True).t()
            #
            final_remission = F.linear(neural_region_corr.t(), pc_remission.t())/per_neural_region_num
            #
            if current_growsp is not None:
                feats = F.linear(neural_region_corr.t(), feats.t()) / per_neural_region_num
                feats = F.normalize(feats, dim=-1)
            #
            for p in torch.unique(neural_region):
                if p!=-1:
                    mask = p==neural_region
                    pfh.append(compute_hist(normals[mask]).unsqueeze(0))

            pfh = torch.cat(pfh, dim=0)
            feats = F.normalize(feats, dim=-1)
            # #
            feats = torch.cat((feats, args.c_rgb*final_remission, args.c_shape*pfh), dim=-1)
            feats = F.normalize(feats, dim=-1)

            point_feats_list.append(feats.cpu())
            point_labels_list.append(labels.cpu())

            all_sp_index.append(neural_region)
            context.append((scene_name, gt, raw_region))

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    return point_feats_list, point_labels_list, all_sp_index, context



def get_pseudo(args, context, cluster_pred, all_sp_index=None):
    print('computing pseduo labels...')
    pseudo_label_folder = args.pseudo_label_path + '/'
    if not os.path.exists(pseudo_label_folder):
        os.makedirs(pseudo_label_folder)
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in range(len(context)):
        scene_name, labels, region = context[i]

        sub_cluster_pred = all_sp_index[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = labels[valid_mask]
        pseudo_gt = -torch.ones_like(labels)
        pseudo_gt_tmp = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(labels.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                pseudo_gt_tmp[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = pseudo_gt_tmp
        #
        pc_no += 1
        new_region = np.unique(sub_cluster_pred)
        region_num += len(new_region[new_region != -1])

        pseudo_label_file = pseudo_label_folder + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    return all_pseudo, all_gt, all_pseudo_gt


def get_pseudo_kitti(args, context, cluster_pred, all_sub_cluster=None):
    print('computing pseduo labels...')
    all_gt = []
    all_pseudo = []
    all_pseudo_gt = []
    pc_no = 0
    region_num = 0

    for i in range(len(context)):
        scene_name, labels, region = context[i]

        sub_cluster_pred = all_sub_cluster[pc_no]+ region_num
        valid_mask = region != -1

        labels_tmp = labels[valid_mask]
        pseudo_gt = -torch.ones_like(labels)
        pseudo_gt_tmp = pseudo_gt[valid_mask]

        pseudo = -np.ones_like(labels.numpy()).astype(np.int32)
        pseudo[valid_mask] = cluster_pred[sub_cluster_pred]

        for p in np.unique(sub_cluster_pred):
            if p != -1:
                mask = p == sub_cluster_pred
                sub_cluster_gt = torch.mode(labels_tmp[mask]).values
                pseudo_gt_tmp[mask] = sub_cluster_gt
        pseudo_gt[valid_mask] = pseudo_gt_tmp
        #
        pc_no += 1
        new_region = np.unique(sub_cluster_pred)
        region_num += len(new_region[new_region != -1])

        pseudo_label_folder = args.pseudo_label_path + '/' + scene_name[0:3]
        if not os.path.exists(pseudo_label_folder):
            os.makedirs(pseudo_label_folder)

        pseudo_label_file = args.pseudo_label_path + '/' + scene_name + '.npy'
        np.save(pseudo_label_file, pseudo)

        all_gt.append(labels)
        all_pseudo.append(pseudo)
        all_pseudo_gt.append(pseudo_gt)

    all_gt = np.concatenate(all_gt)
    all_pseudo = np.concatenate(all_pseudo)
    all_pseudo_gt = np.concatenate(all_pseudo_gt)

    return all_pseudo, all_gt, all_pseudo_gt


def get_fixclassifier(in_channel, centroids_num, centroids):
    classifier = nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
    centroids = F.normalize(centroids, dim=1)
    classifier.weight.data = centroids
    for para in classifier.parameters():
        para.requires_grad = False
    return classifier


def compute_hist(normal, bins=10, min=-1, max=1):
    ## normal : [N, 3]
    normal = F.normalize(normal)
    relation = torch.mm(normal, normal.t())
    relation = torch.triu(relation, diagonal=0) # top-half matrix
    hist = torch.histc(relation, bins, min, max)
    # hist = torch.histogram(relation, bins, range=(-1, 1))
    hist /= hist.sum()

    return hist
