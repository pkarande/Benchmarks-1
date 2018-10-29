from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
# from matplotlib.widgets import Slider, Button, RadioButtons
import csv
import matplotlib as mpl
import time
import hdbscan
import glob
import os
import pandas as pd

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

class node(object):
    def __init__(self, name, parent=None, birth=-1, death=100):
        self.name = name
        self.parent = parent
        # self.children = children
        self.birth = birth
        self.death = death
        self.x = -1
        self.y = -1

    def add_parent(self, parent):
        self.parent = parent

    def set_birth(self, birth):
        self.birth = birth

    def set_death(self, death):
        self.death = death

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Name:{}, Parent:{}, Pos: ({:2.2f}, {:2.2f}), Born:{:2.6f}, Died:{:2.6f}".format(self.name,
                                                                                                self.parent,
                                                                                                self.x, self.y,
                                                                                                self.birth, self.death)

    def get_dict(self):
        return {'Name': self.name, 'Parent': self.parent,
                'x_pos': self.x, 'y_pos': self.y,
                'Born': self.birth, 'Died': self.death}


def write_talass_csv(clusterer, x_all, y_all, filename):

    # Get condensed tree output
    tree = clusterer.condensed_tree_
    tree_pandas = tree.to_pandas()

    # Modify tree to show ditances instead of lambda
    tree_copy = tree_pandas.copy()
    tree_copy['lambda_val'] = 1. / (tree_pandas['lambda_val'] + 1)
    tree_copy['dist'] = tree_copy['lambda_val']
    tree_copy = tree_copy.drop(['lambda_val'], axis=1)

    # Create a node Dictionary of the tree
    max_distance = 1.025*tree_copy['dist'].max()
    min_distance = tree_copy['dist'].min()
    parent_nodes = []
    node_dict = {}
    for ind, row in tree_copy.iterrows():

        child = node(name=row['child'].astype(int),
                     parent=row['parent'].astype(int),
                     birth=row['dist'],
                     death=max_distance)

        node_dict[str(child.name)] = child
        if row['child_size']>1:
            # print 'Adding to '
            parent_nodes.append(str(child.name))

        if child.name < len(x_all):
            child.set_position(x_all[child.name], y_all[child.name])

    absent_parents = []
    for leaf in node_dict.iterkeys():
        parent = node_dict[leaf].parent
        if str(parent) not in node_dict.keys():
            absent_parents.append(parent)

    for parent in np.unique(absent_parents):
        parent_nodes.append(str(parent))
        parent = node(name=parent.astype(int),
                      parent=-1,
                      birth=max_distance,
                      death=1.05*max_distance)
        node_dict[str(parent.name)] = parent


    absent_parent = np.unique(absent_parents)
    for leaf in node_dict.iterkeys():
        parent = node_dict[leaf].parent
        if parent == absent_parent:
            node_dict[leaf].add_parent(1600)
        if node_dict[leaf].name == absent_parent:
            node_dict[leaf].name = 1600

    for leaf in node_dict.iterkeys():
        parent = node_dict[leaf].parent
        try:
            node_dict[leaf].set_death(node_dict[str(parent)].birth)
        except:
            pass


    # Create a dataframe from node dict
    node_df = pd.DataFrame(columns=['Name', 'Parent', 'x_pos', 'y_pos', 'Born', 'Died'])
    for leaf in node_dict.iterkeys():
        node_df = node_df.append(node_dict[leaf].get_dict(), ignore_index=True)

    # Write to csv
    node_df = node_df.sort_values('Name')
    node_df.to_csv(filename, index=False, header=False)

def normalize_AEfeatures(features):
    scaler = MinMaxScaler()
    norm_feats = scaler.fit_transform(features)

    return norm_feats


'''def toCSV(domains, filename):
    keys = domains[0].keys()
    filename = filename + '-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
    with open(filename, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(domains)

'''

def encodeLabels(lipid_type):
    '''
    Convert Lipid names to int labels
    '''
    encoder = LabelEncoder()
    return encoder.fit_transform(lipid_type)

def flip_weights(weights):
    sign = np.sign(weights)
    weights = np.abs(weights)
    weights = np.divide(1., weights)
    weights = weights / sum(weights)
    weights = sign * weights
    return weights

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


def get_ae_file_frame(frame_num, ae_files):
    # ae_index = int(frame_file.split('-f')[-1].split('.npz')[0])
    # print 'Frame number: ', ae_index
    ae_index = int(frame_num)
    ae_index += 100
    ae_file = [s for s in ae_files if 'chunk_' + str(ae_index)[:2] in s]
    frame_ind = int(str(ae_index)[2:])

    return frame_ind, ae_file

def get_nbr_file_frame(run_file, frame_num, nbr_dir):
    run_num = os.path.basename(run_file).split('.')[0][:-7]
    print 'Run num:', run_num
    run_dir = glob.glob(nbr_dir + '/*' + run_num + '*')[0]
    # print 'Frame number: ', ae_index
    nbr_index = int(frame_num)
    nbr_index += 100
    nbr_file = glob.glob(run_dir + '/*chunk_' + str(nbr_index)[:2] + '*')[0]
    frame_ind = int(str(nbr_index)[2:])

    return frame_ind, nbr_file

def get_nbr_file(run_num, frame_num, nbr_dir):
    # run_num = os.path.basename(md_run).split('.')[0][:-7]
    run_dir = glob.glob(nbr_dir + '*' + run_num + '*')[0]
    ae_index = int(frame_num)
    ae_index += 100
    nbr_file = glob.glob(run_dir + '/*chunk_' + str(ae_index)[:2] + '*')[0]
    return nbr_file


def get_leaflet_file(run_num, frame_num, leaflet_dir):
    # run_num = os.path.basename(md_run).split('.')[0][:-7]
    run_dir = glob.glob(leaflet_dir + '*' + run_num + '*')[0]
    ae_index = int(frame_num)
    ae_index += 100
    nbr_file = glob.glob(run_dir + '/*chunk_' + str(ae_index)[:2] + '*')[0]
    return nbr_file

def get_lipid_com(x):
    com = np.zeros((len(x), 2))
    for i in range(len(x)):
        head_inds = np.argwhere(x[i, :, 6] == 1)
        com[i] = np.mean(x[i, head_inds, :2], axis=0)
    return com

def get_data_arrays(f):

    data = np.load(f)
    X = data['features']
    nbrs = data['neighbors']
    resnums = data['resnums']

    return (X, nbrs, resnums)

def get_md_run(ae_epochs):
    aefiles = glob.glob(ae_epochs[0]+'/3k*npy')[0]
    md_run = os.path.basename(aefiles).split('_')[1]
    return md_run
