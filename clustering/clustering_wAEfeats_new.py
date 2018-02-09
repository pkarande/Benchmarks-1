import glob
import clusteringHelper as hf
import hullfunction as hull
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np
import optparse
import time
import seaborn as sns


def main(frame_num, aeFileDir, leafletDir, featureDir, epsilon, num_samples, periodic, leaflet, thresh, hullAlpha, show_plot):
    '''
    Main function
    '''

    aeFiles = glob.glob(aeFileDir+'3k*_outof_*_AE_*npy')
    leafletFiles = glob.glob(leafletDir+'3k*leaflet*_outof_*npz')
    featureFiles = glob.glob(featureDir+'3k*_outof_*npz')

    frame_ind, ae_file, leaflet_file, feature_file = hf.get_files_from_frame_num(frame_num,
                                                                                 aeFiles,
                                                                                 leafletFiles,
                                                                                 featureFiles)

    print "frame_ind: ", frame_ind
    print "ae_file: \n", ae_file[0]
    print "leaflet_file: \n", leaflet_file[0]
    print "feature_file: \n", feature_file[0]

    aeFeats = np.load(ae_file[0])
    data = np.load(feature_file[0])
    leaflet_data = np.load(leaflet_file[0])['leaflet']

    if leaflet == "outer":
        lipid_id = np.argwhere(np.squeeze(leaflet_data[frame_ind, :, :]) == 1)
        lipid_id = np.squeeze(lipid_id.astype(int))
    elif leaflet == "inner":
        lipid_id = np.argwhere(np.squeeze(leaflet_data[frame_ind, :, :]) == 2)
        lipid_id = np.squeeze(lipid_id.astype(int))

    # Extract features and Creat lables
    features = data['features'][frame_ind, lipid_id, :, :]
    features = np.squeeze(features)

    oneHot_labels = features[:, 0, 3:6]
    lipid_labels = np.argmax(oneHot_labels, axis=1).astype(int)
    '''
    print 'lipid_id shape: ', lipid_id.shape
    print 'dim shape ', dim
    print 'features shape: ', features.shape
    print 'Onehot shape: ', oneHot_labels.shape
    print 'lipid_label shape: ', lipid_labels.shape'''

    features = np.float_(features)

    # Extract AE features
    aeFrame = np.squeeze(aeFeats)
    aeFrame = aeFrame[frame_ind, :, :]
    aeFrame = hf.normalize_AEfeatures(aeFrame)
    aeFrame = aeFrame[lipid_id, :]

    xy_feats = features[:, 0, :2]

    dim = np.max(features[:, 0, :3], axis=0)
    dim = np.divide(dim, np.max(dim))
    dim_xy = dim[:2]
    # dim = np.multiply(weights[:3], dim)

    x = xy_feats[:, 0]
    y = xy_feats[:, 1]

    x_all = np.hstack([x, x, x+dim_xy[0], x+dim_xy[0]])
    y_all = np.hstack([y, y+dim_xy[1], y+dim_xy[1], y])

    x_all = np.divide(x_all, np.max(dim_xy))
    y_all = np.divide(y_all, np.max(dim_xy))

    # For now no hull
    x_all = x
    y_all = y

    print 'lipid_id shape: ', lipid_id.shape
    print 'dim shape ', dim
    print 'features shape: ', features.shape
    print 'ae features shape: ', aeFrame.shape
    print 'Onehot shape: ', oneHot_labels.shape
    print 'lipid_label: ', lipid_labels

    # Get distance matrix
    aeWeights = np.ones(aeFrame.shape[1])
    total = hf.computePairWiseDistance(aeFrame, weights=aeWeights,
                                       cube=dim, isPeriodic=periodic)

    # Calculate DBSCAN fit
    if num_samples < 5:
        print "Minimum points parameter cant be <5... Setting to 5"
        num_samples = 5

    if thresh < 5:
        print "Minimum points in a cluster cant be <5... Setting to 5"
        thresh = 5

    fit = hf.findClusters(total, epsilon=epsilon,
                          num_samples=num_samples, thresh=thresh)

    pred_label = fit.labels_
    pred_all = np.hstack([pred_label, pred_label, pred_label, pred_label])
    pred_all = pred_label
    # Set colormap
    # c_map = 'gist_earth'
    c_map = ListedColormap(sns.color_palette("tab20").as_hex())

    # Function to create Hull
    def createHull(pred_all, splot, hullAlpha=0.4):
        clust_labels = pred_all
        clust_num, clust_count = np.unique(
            clust_labels[clust_labels != -1], return_counts=True)
        sorted_clusters = np.argsort(clust_count)

        plot_labels = -1*np.ones_like(clust_labels)

        for i in sorted_clusters:

            color = eval('mpl.cm.' + c_map +
                         '(' + str(float(i) / len(sorted_clusters)) + ')')
            color = color[:3]

            data_with_clust = np.hstack([x_all[clust_labels == i].reshape(-1, 1),
                                        y_all[clust_labels == i].reshape(-1, 1),
                                        pred_all[clust_labels == i].reshape(-1, 1)])

            if len(data_with_clust) < 3:
                continue

            concave_hull, edge_points = hull.alpha_shape(data_with_clust, hullAlpha)

            try:
                num_poly = len(concave_hull.boundary)

                hull_areas = np.zeros((num_poly,))
                for n in range(num_poly):
                    hull_areas[n] = concave_hull[n].area
                n = np.argmax(hull_areas)
                to_plot = n
                polygon = np.array(concave_hull.boundary[n])

                if polygon.shape[0] < 5:
                    for p in range(len(polygon)):
                        x = np.hstack([x_all.reshape(-1, 1), y_all.reshape(-1, 1)]) == polygon[p]
                        plot_labels[x[:, 0]] = -1
                    continue

                path = mpl.path.Path(polygon)
                inside = path.contains_points(np.hstack([x_all.reshape(-1, 1), y_all.reshape(-1, 1)]))
                plot_labels[inside] = i

            except:
                polygon = np.array(concave_hull.boundary)
                if polygon.shape[0] < 5:
                    plot_labels[plot_labels == i] = -1
                    continue

                path = mpl.path.Path(polygon)
                inside = path.contains_points(np.hstack([x_all.reshape(-1, 1), y_all.reshape(-1, 1)]))
                plot_labels[inside] = i

            # print "print concave hull..."

            patch = PolygonPatch(concave_hull[to_plot], fc='#CCCCCC', ec='#553343',
                                 fill=True, zorder=-1, linewidth=1.0, color='b')
            splot.add_patch(patch)
            splot.scatter(x_all[plot_labels == i], y_all[plot_labels == i], c=color, s=9)

        splot.scatter(x_all[plot_labels == -1],
                      y_all[plot_labels == -1],
                      s=3, c='k', alpha=.5, marker='.')

    # Plot the data using lipid_lables

    fig = plt.figure(figsize=(12, 7))
    splot1 = fig.add_subplot(121)
    splot1.scatter(x_all, y_all,
                   c=lipid_labels, s=18, alpha=1, cmap=c_map)
    splot1.axis('square')
    splot1.axis('off')
    plt.title('Labeled lipids')

    # Plot the data using fit_labels

    splot2 = fig.add_subplot(122)
    # Plotting detected clusters

    pred_all = -1*pred_all
    pred_all = pred_all - np.min(pred_all)
    print '\nNumber of clusters: ', len(np.unique(pred_label))
    clusters = splot2.scatter(x_all[pred_all > -1], y_all[pred_all > -1],
                              c=pred_all[pred_all > -1], s=18, alpha=1, cmap=c_map)
    # Plotting noisy samples
    noise = splot2.scatter(x_all[pred_all == -1], y_all[pred_all == -1],
                           s=6, c='k', alpha=.1, marker='.')
    splot2.axis('square')
    splot2.axis('off')
    plt.title('Detected AE clusters')
    # plt.subplots_adjust(left=0.01, right=.99, wspace=.05, top=1)
    ax2 = plt.gca()

    # Plot clusters with concave hull
    '''
    splot3 = fig.add_subplot(133)
    #createHull(pred_all, splot3, hullAlpha)
    splot3.axis('square')
    splot3.axis('off')
    plt.title('Clusters with boundaries')
    #plt.subplots_adjust(left=0.01, right=.99, wspace=.05, top=1)
    ax3 = plt.gca()
    '''
    # Adjust layout
    plt.tight_layout()
    '''
    weight_str = np.array2string(weights, precision=2, separator=', ')
    weight_text = fig.text(.05, .90, 'Weights: ' + weight_str)
    weight_text.set_visible(False)'''

    epsInit = epsilon
    ptsInit = num_samples
    # alphaInit = hullAlpha
    sizeInit = thresh
    axcolor = 'lightgoldenrodyellow'

    axPts = plt.axes([0.25, 0.10, 0.6, 0.03], facecolor=axcolor)
    axEps = plt.axes([0.25, 0.06, 0.6, 0.03], facecolor=axcolor)
    axSize = plt.axes([0.25, 0.02, 0.6, 0.03], facecolor=axcolor)
    # axHullAlpha = plt.axes([0.25, 0.025, 0.6, 0.03], facecolor=axcolor)
    axSave = plt.axes([.85, .9, .1, .05], facecolor=axcolor)

    sEps = Slider(axEps, 'Epsilon', 0.0, 1.0, valinit=epsInit, valfmt='%.2f')
    sPts = Slider(axPts, 'Minimum Pts.', 5, 20, valinit=ptsInit, valfmt='%i')
    # sAlpha = Slider(axHullAlpha, 'Hull Alpha', 0.0, 1.0,
    #                valinit=alphaInit, valfmt='%.2f')
    sSize = Slider(axSize, 'Cluster Size', 5, 20,
                   valinit=sizeInit, valfmt='%i')
    saveButton = Button(axSave, 'Save')

    # Functions to update plot using sliders #####
    def update(val):

        ax2.clear()
        fit = hf.findClusters(total, epsilon=sEps.val,
                              num_samples=np.int(sPts.val), thresh=np.int(sSize.val))

        pred_label = fit.labels_
        pred_all = np.hstack([pred_label, pred_label, pred_label, pred_label])
        pred_all = pred_label

        pred_all = -1*pred_all
        pred_all = pred_all - np.min(pred_all)
        print '\nNumber of clusters: ', len(np.unique(pred_label))
        splot2.scatter(x_all[pred_all > -1], y_all[pred_all > -1],
                       c=pred_all[pred_all > -1], s=18, alpha=1, cmap=c_map)

        splot2.scatter(x_all[pred_all == -1], y_all[pred_all == -1],
                       s=6, c='k', alpha=.1, marker='.')

        splot2.axis('square')
        splot2.axis('off')
        splot2.set_title('Detected AE clusters')
        '''
        ax3.clear()
        createHull(pred_all, splot3, sAlpha.val)
        splot3.axis('square')
        splot3.axis('off')
        splot3.set_title('Clusters with boundaries')
        '''
        fig.canvas.draw_idle()

    def save_fig(event):
        '''
        weight_str = np.array2string(weights_on_plot, precision=2, separator=', ')
        weight_text.set_text('Weights: ' + weight_str)
        weight_text.set_visible(True)
        saved_plots = glob.glob(filename[:-4] + '*.png')

        if not saved_plots:
            fig_loc = filename[:-4] + '-AE-001.png'
            print '\nSaving to: ', fig_loc
            plt.savefig(fig_loc)
        else:
            next_plot_num = int(saved_plots[-1][-7:-4]) + 1
            fig_loc = filename[:-4] + '-AE-' + str(next_plot_num).zfill(3) + '.png'
            print '\nSaving to: ', fig_loc
            plt.savefig(fig_loc)
        '''
        name_str = ae_file[0][:-4] + '-' + time.strftime("%Y%m%d-%H%M%S") + '.png'
        plt.savefig(name_str)

    # Update plots using sliders
    sEps.on_changed(update)
    sPts.on_changed(update)
    sSize.on_changed(update)
    # sAlpha.on_changed(update)
    saveButton.on_clicked(save_fig)

    if show_plot:
        plt.show()
    else:
        save_fig(1)

    return


if __name__ == "__main__":

    parser = optparse.OptionParser()

    parser.add_option("--frame",
                      dest="frame_num",
                      type=int, metavar="frame_num",
                      default=1430,
                      help="number of the frame to cluster on")

    parser.add_option("--aeDir",
                      dest="aeFileDir",
                      type=str, metavar="aeFileDir",
                      default="run10/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20_chunk_16_outof_29_AE_IncludeFalse_ConvFalse.npy",
                      help="directory with npz files of latent space data from the autoencoder")

    parser.add_option("--leafletDir",
                      dest="leafletFileDir",
                      type=str, metavar="leafletFileDir",
                      help="directory with npz files of leaflet data")

    parser.add_option("--featDir",
                      dest="featFileDir",
                      type=str, metavar="featFileDir",
                      help="directory with npz files of features used to train the autoencoder")

    parser.add_option("-p", "--periodic",
                      action="store_true",
                      dest="periodic", default=False,
                      help="sets periodic boundaries in x,y,z")

    parser.add_option("-e", "--eps",
                      dest="epsilon", metavar="epsilon",
                      default=0.146, type=float,
                      help="eps parameter for DBSCAN")

    parser.add_option("-n", "--n_samples",
                      dest="num_samples", metavar="num_samples",
                      default=5, type=int,
                      help="min_sample parameter for DBSCAN, has to be >=5")

    parser.add_option("--leaflet",
                      dest="leaflet", metavar="leaflet",
                      default="outer", type=str,
                      help="Select the leaflet in current frame")

    parser.add_option("--thresh",
                      dest="thresh", metavar="thresh",
                      default=5, type=int,
                      help="minimum number of points in cluster, must be >=5")

    parser.add_option("--hull",
                      dest="hullAlpha", metavar="hullAlpha",
                      default=0.4, type=float,
                      help="alpha value to influence the gooeyness of the border, [0. 1.]")

    parser.add_option("--plot",
                      action="store_true",
                      dest="show_plot", default=False,
                      help="Produce an interactive plot")

    (options, args) = parser.parse_args()

    print("Running... \nFrame num: {}\nAE Dir: {} \nLeaflet Dir: {} \nFeature Dir: {} \nPeriodic: {}\nEpsilon: {}\nNum_samples: {}\nLeaflet: {}\nThreshold: {}\nHull Alpha: {}\nShow plot: {}\n\n"
          .format(options.frame_num, options.aeFileDir, options.leafletFileDir, options.featFileDir, options.periodic, options.epsilon, options.num_samples, options.leaflet, options.thresh, options.hullAlpha, options.show_plot))

    main(options.frame_num, options.aeFileDir, options.leafletFileDir, options.featFileDir,
         options.epsilon, options.num_samples, options.periodic, options.leaflet,
         options.thresh, options.hullAlpha, options.show_plot)
