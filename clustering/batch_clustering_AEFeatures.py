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

    # Read directories with different training runs
    ae_train_runs = np.sort(glob.glob('../Pilot2/P2B1/molecular_AE_results/09202018_*'))
    print 'Traing runs to cluster on:\n', ae_train_runs

    # Loop through each directory
    for aeTrainDir in ae_train_runs:

        print '\n\nTraining run..\n', aeTrainDir

        outpath = aeTrainDir + '/results_new'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        ae_epochs = np.sort(glob.glob(aeTrainDir + '/epoch*'))
        try:
            md_run = ch.get_md_run(ae_epochs)
        except Exception as e:
            print 'Skipping run: ', aeTrainDir
            continue

        print 'md_run: ', md_run
        model_file = glob.glob(aeTrainDir + '/*txt')[0]

        # Loop through each epoch in the directory and perform clustering
        for aeFileDir in ae_epochs:

            try:
                aefiles = glob.glob(aeFileDir+'/3k*npy')
            except Exception as e:
                print '\nNo saved data for {}, skipping.. \n'.format(aeFileDir)
                continue

            frame_ind, ae_file = ch.get_ae_file_frame(frame_num, aefiles)

            try:
                aeFeats = np.load(ae_file[0])
                print '\nClustering from...\n{}\nBlock frame num: {}'.format(aefiles[0], frame_ind)
            except Exception as e:
                print '\nNo saved data for frame {}, skipping.. \n'.format(frame_num)
                continue

            # Get the leaflet of each molecule
            leaflet_dir = '/p/gscratchr/brainusr/datasets/cancer/pilot2/3k_leaflets/'
            leaflet_file = ch.get_leaflet_file(md_run, frame_num, leaflet_dir)
            print 'Getting leaflet from..\n', leaflet_file
            leaflet_data = np.load(leaflet_file)['leaflet']
            leaflet_data = leaflet_data[frame_ind, :, 0]

            if leaflet == "outer":
                lipid_id = np.argwhere(leaflet_data == 1).reshape(-1)
                lipid_id = lipid_id.astype(int)
            elif leaflet == "inner":
                lipid_id = np.argwhere(leaflet_data == 2).reshape(-1)
                lipid_id = lipid_id.astype(int)
            else:
                print("Error - Leaflet can only be \"inner\" or \"outer\"")


            # get neighbors of each molecule
            nbr_dir = '/p/gscratchr/brainusr/datasets/cancer/pilot2/'
            nbr_file = ch.get_nbr_file(md_run, frame_num, nbr_dir)
            print 'Getting neighbors from:\n', nbr_file
            (X, nbrs_all, resnums) = ch.get_data_arrays(nbr_file)
            X_frame = X[frame_ind, lipid_id, :, :]
            nbrs_frame = nbrs_all[frame_ind, :, :num_nbrs].astype(int)

            # Create lables
            lipid_labels = X_frame[:, 0, 3:6]
            lipid_labels = np.argmax(lipid_labels, axis=1)

            # Extract AE features
            aeFrame = np.squeeze(aeFeats)
            aeFrame = aeFrame[frame_ind, :, :]
            aeFrame = ch.normalize_AEfeatures(aeFrame)
            aeFrame_all = np.copy(aeFrame)
            aeFrame = aeFrame[lipid_id, :]

            xy_feats = ch.get_lipid_com(X_frame)

            dim_xy = np.max(xy_feats, axis=0)

            x_all = xy_feats[:, 0]
            y_all = xy_feats[:, 1]

            '''x_all = np.hstack([x, x, x+dim_xy[0], x+dim_xy[0]])
            y_all = np.hstack([y, y+dim_xy[1], y+dim_xy[1], y])

            x_all = np.divide(x_all, np.max(dim_xy))
            y_all = np.divide(y_all, np.max(dim_xy))

            x_all = x
            y_all = y'''

            # Calculate sparse distance matrix including only the neighbouring molecules
            dist = np.ones((3040, 3040))*np.inf
            dist_all = pairwise_distances(aeFrame_all)
            for i in range(3040):
                dist[i ,nbrs_frame[i, :]] = dist_all[i, nbrs_frame[i, :]]

            dist = dist[lipid_id]
            dist = dist[:, lipid_id]


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
            fig_name = md_run + '_' +\
                       os.path.basename(model_file).split('_')[1] + '_' +\
                       os.path.basename(aeFileDir) + '.png'
            plt.savefig(outpath + '/' + fig_name)
            plt.close('all')


            # Wrtie the HDBSCAN output in a format that can be read by Talass
            talass_file_name = md_run + '_' +\
                               os.path.basename(model_file).split('_')[1] + '_' +\
                               os.path.basename(aeFileDir) + '.csv'

            ch.write_talass_csv(clusterer, x_all, y_all, outpath + '/' + talass_file_name)


if __name__ == '__main__':
    main()
