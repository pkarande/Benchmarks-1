from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from matplotlib.widgets import Slider, Button, RadioButtons
import csv
import matplotlib as mpl
import time

'''
Comment the following line if there are issues with
displaying matplotlib plots
'''
if "MACOSX" in mpl.get_backend().upper():
    print 'Changing backend to TkAgg'
    mpl.use("TkAgg")
if "module://ipykernel.pylab.backend_inline" in mpl.get_backend():
    print 'Changing backend to nbagg'
    mpl.use("nbagg")

import matplotlib.pyplot as plt
import numpy as np


def flip_weights(weights):
    sign = np.sign(weights)
    weights = np.abs(weights)
    weights = np.divide(1., weights)
    weights = weights / sum(weights)
    weights = sign * weights
    return weights


def periodicDistance(x0, x1, dimensions):
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))


def computePairWiseDistance(features, weights, cube=[1, 1, 1],
                            periodicFeats=np.array([[1, 2, 3], [4, 5, 6]]),
                            isPeriodic=True):
    '''
    Function to return pairwise distances
    features: number of samples X features
    periodicFeats: 2d numpy array of periodic featuers
    isPeriodic: flag for periodic cartesian boundaries
    '''

    if isPeriodic:
        geo_dist = np.zeros([features.shape[0], features.shape[0]])

        for p_feat in periodicFeats:

            nonzero_p_feat = np.argwhere(weights[p_feat] != 0)
            nonzero_p_feat = np.squeeze(nonzero_p_feat)

            p_feat = p_feat[nonzero_p_feat]

            if len(p_feat) == 0:
                continue

            xyz_feats = features[:, p_feat]
            xyz_feats_rep = np.expand_dims(xyz_feats, axis=1)
            xyz_feats_rep = np.repeat(xyz_feats_rep,
                                      repeats=xyz_feats.shape[0],
                                      axis=1)

            dim = np.multiply(cube[nonzero_p_feat], weights[p_feat])

            geo_dist += periodicDistance(xyz_feats_rep,
                                         xyz_feats,
                                         dim)**2

        non_periodic_feats = np.setdiff1d(range(features.shape[1]), periodicFeats)
        non_periodic_feats = np.setdiff1d(non_periodic_feats, np.where(weights == 0))

        other_dist = pdist(features[:, non_periodic_feats])
        other_dist = squareform(other_dist)

        distance = np.sqrt(geo_dist + other_dist**2)
    else:

        distance = pdist(features[:, weights != 0])
        distance = squareform(distance)
    # Normalizing distances
    distance = distance * 2 / np.max(distance)
    return distance


def findClusters(features, epsilon, num_samples, dist_metric='precomputed', thresh=5):
    '''
    Function to find clusters using DBSCAN
    Returns DBSCAN.fit()
    '''
    clust = DBSCAN(eps=epsilon, min_samples=num_samples, metric=dist_metric)
    fit = clust.fit(features)

    labels, count = np.unique(fit.labels_, return_counts=True)
    small_clust = labels[count <= thresh]
    for lab in small_clust:
        fit.labels_[fit.labels_ == lab] = -1

    return fit


def OneHot(lipid_type):
    '''
    Convert Lipid Labels to one hot encoded labels
    '''

    encoder = LabelBinarizer()
    return encoder.fit_transform(lipid_type)


def encodeLabels(lipid_type):
    '''
    Convert Lipid names to int labels
    '''
    encoder = LabelEncoder()
    return encoder.fit_transform(lipid_type)


def normalize_features(features, dim, weights, head_xyz=[1, 2, 3], tail_xyz=[4, 5, 6]):
    norm_feats = np.zeros(features.shape)

    norm_feats[:, head_xyz] = np.divide(features[:, head_xyz], np.max(dim))
    norm_feats[:, tail_xyz] = np.divide(features[:, tail_xyz], np.max(dim))

    feat_num = range(features.shape[1])
    feat_num = np.setdiff1d(feat_num, head_xyz)
    feat_num = np.setdiff1d(feat_num, tail_xyz)

    scaler = MinMaxScaler()
    temp = scaler.fit_transform(features[:, feat_num])

    norm_feats[:, feat_num] = temp
    feat_scale = np.array(
        [np.min(norm_feats, axis=0), np.max(norm_feats, axis=0)])

    # flip featuers with negative weights
    head_tail = np.append(head_xyz, tail_xyz)
    opp_features = np.argwhere(weights < 0)
    opp_head_tail = np.intersect1d(opp_features, head_tail)

    norm_feats[:, opp_features] = np.abs(norm_feats[:, opp_features] - 1)

    for feat in opp_head_tail:
        scaler = MinMaxScaler(feature_range=feat_scale[:, feat])
        norm_feats[:, feat] = np.squeeze(
            scaler.fit_transform(norm_feats[:, feat].reshape(-1, 1)))

    return norm_feats


def normalize_AEfeatures(features):
    scaler = MinMaxScaler()
    norm_feats = scaler.fit_transform(features)

    return norm_feats


def toCSV(domains, filename):
    keys = domains[0].keys()
    filename = filename + '-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
    with open(filename, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(domains)


def sel_data(data, leaflet):

    if leaflet == 'outer':
        features = data['featureOuter'][:, 3:]
        lipid_type = data['featureOuter'][:, 0]
        lipid_id = data['featureOuter'][:, 1]
    elif leaflet == 'inner':
        features = data['featureInner'][:, 3:]
        lipid_type = data['featureInner'][:, 0]
        lipid_id = data['featureInner'][:, 1]
    else:
        print 'Leaflet can only be \'inner\' or \'outer\''

    dim = data['dim']
    features = np.float_(features)
    return lipid_id, lipid_type, dim, features


def get_ae_file_frame(frame_file, ae_files):
    ae_index = int(frame_file.split('-f')[-1].split('.npz')[0])
    print ae_index
    ae_index += 100
    ae_file = [s for s in ae_files if 'chunk_' + str(ae_index)[:2] in s]
    frame_ind = int(str(ae_index)[2:]) + 1

    return frame_ind, ae_file


def get_files_from_frame_num(frame_num, ae_Files, leaflet_Files, feature_Files):
    '''ae_index = int(frame_file.split('-f')[-1].split('.npz')[0])
    '''
    ae_index = frame_num
    print ae_index

    ae_index += 100
    ae_file = [s for s in ae_Files if 'chunk_' + str(ae_index)[:2] in s]
    leaflet_file = [s for s in leaflet_Files if 'chunk_' + str(ae_index)[:2] in s]
    feature_file = [s for s in feature_Files if 'chunk_' + str(ae_index)[:2] in s]

    frame_ind = int(str(ae_index)[2:]) + 1

    return frame_ind, ae_file, leaflet_file, feature_file
