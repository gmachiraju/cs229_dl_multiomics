# create an AE and fit it with our data using 3 neurons in the dense layer
# using keras' functional API
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from umap import UMAP <--- dependency clash between UMAP and MOFA....
# ^can look into and fix later

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mofapy import *
from mofapy.core.entry_point import entry_point
from oi_vae.multiomics_oivae import OivaeOmics
import pdb
import h5py
import os
import pickle

# maui stuffs
import urllib.request
from sklearn import preprocessing
import matplotlib.lines as mlines
import seaborn as sns
import maui
import maui.utils
from keras import backend as K

from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# home_dir = "/scratch/users/gmachi/dl_multiomics/cs229_dl_multiomics"


# Not yet implemented:
#---------------------
# run sparse pca from scikitlearn!
def run_sparse_pca(X, encoding_dim=10):
    pass


# run sparse pca from scikitlearn!
def run_kernel_pca(X, encoding_dim=10):
    pass


# run sparse pca from scikitlearn!
def run_mds(X, encoding_dim=10):
    pass


# run sparse pca from scikitlearn!
def run_icluster(X, encoding_dim=10):
    pass


def run_pca(X, encoding_dim=10):
    pca = PCA(n_components=encoding_dim)
    X_embedded = pca.fit_transform(X)
    X_loadings = pca.components_
    return X_loadings, X_embedded


def run_fa(X, encoding_dim=10):
    fa = FactorAnalysis(n_components=encoding_dim, random_state=0)
    X_embedded = fa.fit_transform(X)
    X_loadings = fa.components_
    return X_loadings, X_embedded


# add dropout layer!
def run_vanilla_autoencoder(X, activation, encoding_dim=10):

    # add dropout layers here? set bias = FALSE here?
    input_dim = X.shape[1]
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation=activation)(input_img)
    decoded = Dense(input_dim, activation=activation)(encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    # print(autoencoder.summary()) #<-- shows number of params etc.

    # change epochs to 1000 when doing real test
    history = autoencoder.fit(X, X,
                              epochs=300,
                              batch_size=50,
                              shuffle=True,
                              validation_split=0.1,
                              verbose=0)

    # plot loss
    #-----------
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model train vs validation loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()

    # use our encoded layer to encode the training input (add no dropout here)
    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    X_embedded = encoder.predict(X)
    X_loadings = encoder.get_weights()[0]

    return X_loadings, X_embedded


def run_mofa(X, args, encoding_dim=10):
    """
    Similar to Principal Component Analysis and other latent variable models, this is a hard question to answer.
    It depends on the data set and the aim of the analysis. As a general rule, the bigger the data set, the higher
    the number of factors that you will retrieve, and the less the variance that will be explained per factor.
    If you want to get an overview on the major sources of variability then use a small number of factors (K<=15).
    If you want to capture small sources of variability, for example to do imputation or eQTL mapping, then go for
    a large number of factors (K>25).
    """

    outfile = home_dir + "/" + args.dataset + ".hdf5"

    if args.cached_mfa == False:
        print("hi, cached mofa")
        # Load the data
        # Be careful to use the right delimiter, and make sure that you use the
        # right arguments from pandas.read_csv to load the row names and column
        # names, if appropriate.

        if args.dataset == "ipop":
            M = 5  # Number of views (omics?)
            data = [None] * M
            data[0] = X.loc[:, X.columns.str.startswith(
                'rnaseq')].astype(pd.np.float32)
            data[1] = X.loc[:, X.columns.str.startswith(
                'prot')].astype(pd.np.float32)
            data[2] = X.loc[:, X.columns.str.startswith(
                'metab')].astype(pd.np.float32)
            data[3] = X.loc[:, X.columns.str.startswith(
                'cytok')].astype(pd.np.float32)
            data[4] = X.loc[:, X.columns.str.startswith(
                'clinic')].astype(pd.np.float32)

        elif args.dataset == "motrpac":
            M = 3
            data = [None] * M
            data[0] = X.loc[:, X.columns.str.startswith(
                'proteomics')].astype(pd.np.float32)
            data[1] = X.loc[:, X.columns.str.startswith(
                'proteomics_ph')].astype(pd.np.float32)
            data[2] = X.loc[:, X.columns.str.startswith(
                'untar_met')].astype(pd.np.float32)

        elif args.dataset == "arabidop1":
            M = 2
            data = [None] * M
            data[0] = X.loc[:, X.columns.str.startswith(
                'diff')].astype(pd.np.float32)
            data[1] = X.loc[:, X.columns.str.startswith(
                'srna')].astype(pd.np.float32)

        elif args.dataset == "arabidop2":
            M = 3
            data = [None] * M
            data[0] = X.loc[:, X.columns.str.startswith(
                'diff')].astype(pd.np.float32)
            data[1] = X.loc[:, X.columns.str.startswith(
                'srna')].astype(pd.np.float32)
            data[2] = X.loc[:, X.columns.str.startswith(
                'methyl')].astype(pd.np.float32)

        elif args.dataset == "tcga":
            M = 3
            data = [None] * M
            data[0] = X.loc[:, X.columns.str.startswith(
                'gex')].astype(pd.np.float32)
            data[1] = X.loc[:, X.columns.str.startswith(
                'mut')].astype(pd.np.float32)
            data[2] = X.loc[:, X.columns.str.startswith(
                'cnv')].astype(pd.np.float32)

        # Initialise entry point
        ep = entry_point()

        # Set data
        ep.set_data(data)

        ## Set model options ##
        # factors: number of factors. By default, the model does not automatically learn the number of factors.
        #   If you want the model to do this (based on a minimum variance explained criteria), set `TrainOptions$dropFactorThreshold` to a non-zero value.
        # likelihoods: list with the likelihood for each view. Usually we recommend:
        #   - gaussian for continuous data
        #   - bernoulli for binary data
        #   - poisson for count data
        #   If you are using gaussian likelihoods, we recommend centering the data (specified in data_options) and setting learnIntercept to False.
        #   However, if you have non-gaussian likelihoods, learning an intercept factor is important
        # sparsity: boolean indicating whether to use sparsity.
        # This is always recommended, as it will make the loadings more
        # interpretable.
        if args.dataset == "ipop":
            ep.set_model_options(factors=encoding_dim, likelihoods=[
                "gaussian", "gaussian", "gaussian", "gaussian", "gaussian"], sparsity=True)
        elif args.dataset == "motrpac":
            ep.set_model_options(factors=encoding_dim, likelihoods=[
                "gaussian", "gaussian", "gaussian"], sparsity=True)
        elif args.dataset == "arabidop1":
            ep.set_model_options(factors=encoding_dim, likelihoods=[
                "gaussian", "gaussian"], sparsity=True)
        elif args.dataset == "arabidop2":
            ep.set_model_options(factors=encoding_dim, likelihoods=[
                "gaussian", "gaussian", "gaussian"], sparsity=True)
        elif args.dataset == "tcga":
            ep.set_model_options(factors=encoding_dim, likelihoods=[
                "gaussian", "gaussian", "gaussian"], sparsity=True)

        ## Set data options ##

        # view_names: list with view names
        # center_features: boolean indicating whether to center the features to zero mean.
        #   This only works for gaussian data. Default is TRUE.
        # scale_views: boolean indicating whether to scale views to have the same unit variance.
        #   As long as the scale differences between the data sets is not too high, this is not required. Default is False.
        # RemoveIncompleteSamples: boolean indicating whether to remove samples that are not profiled in all omics.
        # We recommend this only for testing, as the model can cope with samples
        # having missing assays. Default is False.

        if args.dataset == "ipop":
            ep.set_data_options(view_names=["rnaseq", "prot", "metab", "cytok", "clinic"], center_features=True, scale_views=True,
                                RemoveIncompleteSamples=False)
        elif args.dataset == "motrpac":
            ep.set_data_options(view_names=["prot", "prot_ph", "metab"], center_features=True, scale_views=True,
                                RemoveIncompleteSamples=False)
        elif args.dataset == "arabidop1":
            ep.set_data_options(view_names=["diff", "srna"], center_features=True, scale_views=True,
                                RemoveIncompleteSamples=False)
        elif args.dataset == "arabidop2":
            ep.set_data_options(view_names=["diff", "srna", "methyl"], center_features=True, scale_views=True,
                                RemoveIncompleteSamples=False)
        elif args.dataset == "tcga":
            ep.set_data_options(view_names=["gex", "mut", "cnv"], center_features=True, scale_views=True,
                                RemoveIncompleteSamples=False)
        sample_names = X.index

        # Parse the data (optionally center or scale, do some QC, etc.)
        ep.parse_data()

        # Define training options
        # iter: numeric value indicating the maximum number of iterations.
        #   Default is 1000, we recommend setting this to a large value and using the 'tolerance' as convergence criteria.
        # tolerance: numeric value indicating the convergence threshold based on the change in Evidence Lower Bound (deltaELBO).
        #   For quick exploration we recommend this to be around 1.0, and for a thorough training we recommend a value of 0.01. Default is 0.1
        # dropR2: numeric hyperparamter to automatically learn the number of factors.
        #   It indicates the threshold on fraction of variance explained to consider a factor inactive and automatically drop it from the model during training.
        #   For example, a value of 0.01 implies that factors explaining less than 1% of variance (in each view) will be dropped.
        #   Default is 0, which implies that only factors that explain no variance at all will be removed
        # elbofreq: frequency of computation of the ELBO. It is useful to assess convergence, but it slows down the training.
        # verbose: boolean indicating whether to generate a verbose output.
        # seed: random seed. If None, it is sampled randomly
        ep.set_train_options(iter=1000, tolerance=0.1, dropR2=0.00, elbofreq=1, verbose=args.verbosity, seed=2019) # iter used to be 10 for testing

        # Define prior distributions
        ep.define_priors()

        # Define initializations of variational distributions
        ep.define_init()

        # Parse intercept factor
        ep.parse_intercept()

        # Train the model
        ep.train_model()

        # Save the model
        ep.save_model(outfile)

    # read the model back in -- hacky approach since the MOFA code exports and no
    # returned object that I can see
    filename = outfile
    f = h5py.File(filename, 'r')

    expectations = f[list(f.keys())[1]]

    # data = f[list(f.keys())[0]]
    # features = f[list(f.keys())[2]]
    # intercept = f[list(f.keys())[3]]
    # model_opts = f[list(f.keys())[4]]
    # samples = f[list(f.keys())[5]]
    # training_opts = f[list(f.keys())[6]]
    # training_stats = f[list(f.keys())[7]]

    # d_keys = list(data)
    # f_keys = list(features)
    # i_keys = list(intercept)
    # s_keys = list(samples)
    # e_keys = list(expectations)
    # w_keys = list(expectations.get('SW'))

    X_embedded = expectations.get('Z')[:, :]

    if args.dataset == "ipop":
        W_rnaseq = expectations.get('SW').get('rnaseq')[:, :]
        W_prot = expectations.get('SW').get('prot')[:, :]
        W_metab = expectations.get('SW').get('metab')[:, :]
        W_cytok = expectations.get('SW').get('cytok')[:, :]
        W_clinic = expectations.get('SW').get('clinic')[:, :]
        Ws = [W_rnaseq, W_prot, W_metab, W_cytok, W_clinic]
    elif args.dataset == "arabidop1":
        W_diff = expectations.get('SW').get('diff')[:, :]
        W_srna = expectations.get('SW').get('srna')[:, :]
        Ws = [W_diff, W_srna]
    elif args.dataset == "arabidop2":
        W_diff = expectations.get('SW').get('diff')[:, :]
        W_srna = expectations.get('SW').get('srna')[:, :]
        W_methyl = expectations.get('SW').get('methyl')[:, :]
        Ws = [W_diff, W_srna, W_methyl]
    elif args.dataset == "tcga":
        W_gex = expectations.get('SW').get('gex')[:, :]
        W_mut = expectations.get('SW').get('mut')[:, :]
        W_cnv = expectations.get('SW').get('cnv')[:, :]
        Ws = [W_gex, W_mut, W_cnv]

    X_loadings = np.concatenate(Ws, axis=1)
    f.close()

    return X_loadings, X_embedded


def run_maui(X, args, encoding_dim=30, hidden_dim=1100, epochs=300): # epochs used to be 5 for testing

    if args.cached_vae == False:

        if args.dataset == "ipop":
            rnaseq = X.loc[:, X.columns.str.startswith('rnaseq')].T
            prot = X.loc[:, X.columns.str.startswith('prot')].T
            metab = X.loc[:, X.columns.str.startswith('metab')].T
            cytok = X.loc[:, X.columns.str.startswith('cytok')].T
            clinic = X.loc[:, X.columns.str.startswith('clinic')].T
        elif args.dataset == "arabidop1":
            diff = X.loc[:, X.columns.str.startswith('diff')].T
            srna = X.loc[:, X.columns.str.startswith('srna')].T
        elif args.dataset == "arabidop2":
            diff = X.loc[:, X.columns.str.startswith('diff')].T
            srna = X.loc[:, X.columns.str.startswith('srna')].T
            methyl = X.loc[:, X.columns.str.startswith('methyl')].T
        elif args.dataset == "tcga":
            gex = X.loc[:, X.columns.str.startswith('gex')].T
            mut = X.loc[:, X.columns.str.startswith('mut')].T
            cnv = X.loc[:, X.columns.str.startswith('cnv')].T

        # set 1 cpu core
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

        maui_model = maui.Maui(n_hidden=[hidden_dim], n_latent=encoding_dim, epochs=epochs, batch_size=50)  # n_hidden used to be hidden_dim=10, 1100

        if args.dataset == "ipop":
            X_embedded = maui_model.fit_transform(
                {'rnaseq': rnaseq, 'prot': prot, 'metab': metab, 'cytok': cytok, 'clinic': clinic})  # aka. z
        elif args.dataset == "arabidop1":
            X_embedded = maui_model.fit_transform(
                {'diff': diff, 'srna': srna})  # aka. z
        elif args.dataset == "arabidop2":
            X_embedded = maui_model.fit_transform(
                {'diff': diff, 'srna': srna, 'methyl': methyl})  # aka. z
        elif args.dataset == "tcga":
            X_embedded = maui_model.fit_transform(
                {'gex': gex, 'mut': mut, 'cnv': cnv})  # aka. z

        X_loadings = maui_model.encoder.get_weights()[6]

        # check convergence
        # maui_model.hist.plot()

        # X_embedded = np.matmul(X_loadings, X)
        d = home_dir + "/" + args.dataset
        serialize(X_loadings, d + '_maui_loadings.obj')
        serialize(X_embedded, d + '_maui_transformed.obj')

        return X_loadings, X_embedded

    else:
        d = home_dir + "/" + args.dataset

        # X_embedded = np.matmul(X_loadings, X)
        X_loadings = deserialize(d + '_maui_loadings.obj')
        X_embedded = deserialize(d + '_maui_transformed.obj')
        return X_loadings, X_embedded


    
# # needs work... unstable gradients
# def run_oivae(X, args, encoding_dim=10):

#     # outfile = os.path.dirname(os.path.realpath(__file__)) + "/test_oivae.hdf5"
#     # ^use above if need to write to binary

#     if not args.cached_oivae:

#         oivae = OivaeOmics(e_size=encoding_dim)
#         X_embedded = oivae.fit_transform(X)
#         return X_embedded


# def run_logreg(X_embedded, y, omics_flag):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_embedded, y, test_size=0.2)
#     if omics_flag == "single":
#         logreg = LogisticRegression(
#             random_state=0, solver='lbfgs')
#     elif omics_flag == "multi":
#         logreg = LogisticRegression(
#             random_state=0, solver='lbfgs', multi_class='multinomial')
#     model = logreg.fit(X_train, y_train)
#     y_hats = model.predict(X_test)
#     y_probs = model.predict_proba(X_test)

#     # class imbalance - may need to fix that
#     # ensure MOFA doesn't futz with ordering of samples... shouldn't...
#     return y_hats, y_probs, y_test


# ** needs to be less than/equal to 4 for barnes_hit alg.. The main purpose of t-SNE is visualization of high-dimensional data.
# def run_tsne(X, encoding_dim=10):
#     tsne = TSNE(n_components=encoding_dim)
#     X_embedded = tsne.fit_transform(X)
#     X_loadings = tsne.embedding_
#     return X_loadings, X_embedded

# ** conflict!!
# def run_umap(X, encoding_dim=10):
#     u_map = UMAP(n_components=encoding_dim)
#     X_embedded = u_map.fit_transform(X)
#     return X_embedded
