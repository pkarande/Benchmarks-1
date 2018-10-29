import matplotlib as mpl
mpl.use('agg')
import os
import sys
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
import glob
import clusteringHelper as ch
import matplotlib.pyplot as plt
import numpy as np
import optparse

def arg_parser():
    parser = optparse.OptionParser()

    parser.add_option("-n", "--num_nbrs",
                      dest="num_nbrs",
                      type=int, metavar="num_nbrs",
                      default=20,
                      help="number of neighbors to calculate distances to; max 50")

    parser.add_option("-m", "--min_size",
                      dest="min_size",
                      type=int, metavar="min_size",
                      default=15,
                      help="minimum size of the clusters returned by HDBSCAN")

    parser.add_option("-f", "--frame_num",
                      dest="frame_num",
                      type=int, metavar="frame_num",
                      default=1430,
                      help="frame of simulation on which clustering is performed")

    parser.add_option("-l", "--leaflet",
                      dest="leaflet",
                      type=str, metavar="leaflet",
                      default="outer",
                      help="leaflet to perform clustering on; inner or outer")

    (options, args) = parser.parse_args()

    return options

def main():

    # Parse command line arguments
    options = arg_parser()

    num_nbrs = options.num_nbrs
    min_size = options.min_size
    frame_num = options.frame_num
    leaflet = options.leaflet

    print 'NUM_NBRS: ', num_nbrs
    print 'MIN_SIZE: ', min_size
    print 'FRAME_NUM: ', frame_num
    print 'Leaflet: ', leaflet

    # Read files with hand engineered features
    feature_files = np.sort(glob.glob('run*' + str(frame_num) + '*npz'))
    print 'Traing runs to cluster on:\n', feature_files

    # Loop through each directory
    for run_file in feature_files:

        print '\n\nRun..\n', run_file

        outpath = './handFeatures/results'
        if not os.path.exists(outpath):
            os.makedirs(outpath)

            ##### Load data #####
        data = np.load(run_file)
        lipid_id, lipid_type, dim, features = ch.sel_data(data, leaflet)

        lipid_labels = ch.encodeLabels(lipid_type)
        dim_xy = dim[:2]

        # Define weight vector (1's for now)
        feat_weights = np.ones(features.shape[1])
        weights_on_plot = feat_weights
        weights = feat_weights
            #weighted_features = features

        # Apply weights to feature vectors
        weights = ch.flip_weights(weights)
        feat_weights[feat_weights != 0] = weights

        # Normalize the features
        xy_feats = features[:, 1:3]
        features = ch.normalize_features(features, dim, feat_weights)

        weighted_features = np.multiply(feat_weights, features)
        dim = np.divide(dim, np.max(dim))
        #dim = np.multiply(weights[:3], dim)

        x_all = xy_feats[:, 0]
        y_all = xy_feats[:, 1]

        # Get distance matrix
        dist = ch.computePairWiseDistance(weighted_features, weights=feat_weights,
                                        cube=dim, isPeriodic=True)


        # Perform Clustering in the sparse distance matrix
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, allow_single_cluster=False, metric='precomputed')
        fit = clusterer.fit(dist)

        pred_label = fit.labels_
        # pred_all = np.hstack([pred_label, pred_label, pred_label, pred_label])
        pred_all = pred_label


        c_map = 'Spectral'
        # Plot the data using lipid_lables
        fig = plt.figure(figsize=(10, 5))
        splot1 = fig.add_subplot(121)
        splot1.scatter(x_all, y_all,
                       c=lipid_labels, s=18, alpha=.75, cmap=c_map)
        splot1.axis('square')
        splot1.axis('off')
        plt.title('Labeled lipids')
        ax1 = plt.gca()

        # Plot the data using fit_labels
        splot2 = fig.add_subplot(122)
        # Plotting detected clusters
        #print '\nNumber of clusters: ', len(np.unique(pred_all))
        clusters = splot2.scatter(x_all[pred_all > -1], y_all[pred_all > -1],
                                 c=pred_all[pred_all > -1], s=18, alpha=.75, cmap=c_map)
        # Plotting noisy samples
        noise = splot2.scatter(x_all[pred_all == -1], y_all[pred_all == -1],
                               s=6, c='k', alpha=1, marker='.')
        splot2.axis('square')
        splot2.axis('off')
        plt.title('Detected AE clusters')
        # plt.subplots_adjust(left=0.01, right=.99, wspace=.05, top=1)
        ax2 = plt.gca()

        plt.tight_layout()
        fig_name = os.path.basename(run_file).split('.')[0] + '.png'
        plt.savefig(outpath + '/' + fig_name)
        plt.close('all')


        # Wrtie the HDBSCAN output in a format that can be read by Talass
        talass_file_name = os.path.basename(run_file).split('.')[0] + '_talass.csv'

        ch.write_talass_csv(clusterer, x_all, y_all, outpath + '/' + talass_file_name)


if __name__ == '__main__':
    main()
