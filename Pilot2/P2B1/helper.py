import numpy as np
import glob
from keras.losses import mean_squared_error as mse
from keras.losses import mean_absolute_error as mae


def combined_loss(y_true, y_pred):
    return mse(y_true, y_pred) + 0.01*mae(0, y_pred)


def periodicDistance(x0, x1, dimensions):

    for i in range(len(dimensions)):
        delta = x0[:, :, i] - x1[i]
        delta = np.where(delta > 0.5 * dimensions[i], delta - dimensions[i], delta)
        delta = np.where(delta < - (0.5 * dimensions[i]), delta + dimensions[i], delta)
        x0[:, :, i] = delta
    return x0


def get_com(x):
    if x[0, 3] == 1:
        return np.mean(x[:8, :3], axis=0)
    else:
        return np.mean(x[:, :3], axis=0)


def get_local_files(data_dir="/Users/karande1/Benchmarks/Pilot2/common/generate_datasets"):
    '''
    Load data files from local directory
    '''

    data_files = glob.glob('%s/*.npz' % data_dir)
    filelist = [d for d in data_files if 'AE' not in d]
    filelist = sorted(filelist)
    import pilot2_datasets as p2
    fields = p2.gen_data_set_dict()

    return (filelist, fields)


def get_data_arrays(f):

    data = np.load(f)

    X = data['features']
    nbrs = data['neighbors']
    resnums = data['resnums']

    return (X, nbrs, resnums)


def append_nbrs_relative(x, nbrs, num_nbrs):

    new_x_shape = np.array((x.shape[0], np.prod(x.shape[1:])))
    new_x_shape[1] *= num_nbrs+1
    x_wNbrs = np.zeros(new_x_shape)

    for i in range(len(x)):
        nb_indices = nbrs[i, :num_nbrs+1].astype(int)
        nb_indices = nb_indices[nb_indices != -1]
        temp_mols = x[nb_indices]
        com = get_com(x[i])

        temp_mols = periodicDistance(temp_mols, com, [1, 1, 0.3])
        ind = np.argwhere(temp_mols[:, 1, 3] == 1)
        temp_mols[ind, 8:, :] = 0

        newshape = (1, np.prod(temp_mols.shape))
        temp_mols = np.reshape(temp_mols, newshape)

        x_wNbrs[i, :temp_mols.shape[1]] = temp_mols

    return x_wNbrs


def append_nbrs(x, nbrs, num_nbrs):

    new_x_shape = np.array(x.shape)
    new_x_shape[1] *= num_nbrs+1
    x_wNbrs = np.zeros(new_x_shape)

    for i in range(len(x)):
        nb_indices = nbrs[i, :num_nbrs+1].astype(int)
        nb_indices = nb_indices[nb_indices != -1]

        temp_mols = x[nb_indices]
        newshape = (1, np.prod(temp_mols.shape))
        temp_mols = np.reshape(temp_mols, newshape)

        x_wNbrs[i, :temp_mols.shape[1]] = temp_mols

    return x_wNbrs
