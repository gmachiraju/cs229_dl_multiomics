from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import *
from utils import *

import pandas as pd
import numpy as np
import scipy as sci

# from ggplot import *
# from plotnine import *
# from itertools import chain

import pdb
import os
import argparse
from scipy import stats
from scipy import interp


label_map1 = {'harvest_1': 'H',
              'harvest_2': "H",
              'harvest_3': "H",
              'X0h_1': "0",
              'X0h_2': "0",
              'X0h_3': "0",
              'X1hS_1': "S",
              'X1hS_2': "S",
              'X1hS_3': "S",
              'X12hS_1': "S",
              'X12hS_2': "S",
              'X_12hS_3': "S",
              'X48hS_1': "S",
              'X48hS_2': "S",
              'X48hS_3': "S",
              'X1hSL_1': "SL",
              'X1hSL_2': "SL",
              'X_1hSL_3': "SL",
              'X6hSL_1': "SL",
              'X6hSL_2': "SL",
              'X6hSL_3': "SL",
              'X12hSL_1': "SL",
              'X12hSL_2': "SL",
              'X12hSL_3': "SL",
              'X24hSL_1': "SL",
              'X24hSL_2': "SL",
              'X24hSL_3': "SL",
              'X48hSL_1': "SL",
              'X48hSL_2': "SL",
              'X48hSL_3': "SL"}

timestamp_map1 = {'harvest_1': 'H',
                  'harvest_2': "H",
                  'harvest_3': "H",
                  'X0h_1': "0h",
                  'X0h_2': "0h",
                  'X0h_3': "0h",
                  'X1hS_1': "1h",
                  'X1hS_2': "1h",
                  'X1hS_3': "1h",
                  'X12hS_1': "12h",
                  'X12hS_2': "12h",
                  'X_12hS_3': "12h",
                  'X48hS_1': "48h",
                  'X48hS_2': "48h",
                  'X48hS_3': "48h",
                  'X1hSL_1': "1h",
                  'X1hSL_2': "1h",
                  'X_1hSL_3': "1h",
                  'X6hSL_1': "6h",
                  'X6hSL_2': "6h",
                  'X6hSL_3': "6h",
                  'X12hSL_1': "12h",
                  'X12hSL_2': "12h",
                  'X12hSL_3': "12h",
                  'X24hSL_1': "24h",
                  'X24hSL_2': "24h",
                  'X24hSL_3': "24h",
                  'X48hSL_1': "48h",
                  'X48hSL_2': "48h",
                  'X48hSL_3': "48h"}

label_map2 = {'X0h_1': "0",
              'X24hSL_1': "SL",
              'X6hSL_3': "SL",
              'X6hSL_1': "SL",
              'X48hSL_3': "SL",
              'X0h_3': "0",
              'X24hSL_2': "SL",
              'X0h_2': "0",
              'X48hS_1': "S",
              'X48hSL_1': "SL",
              'X48hS_2': "S",
              'X24hSL_3': "SL",
              'X6hSL_2': "SL",
              'X48hSL_2': "SL",
              'X48hS_3': "S"
              }

timestamp_map2 = {'X0h_1': "0h",
                  'X24hSL_1': "24h",
                  'X6hSL_3': "6h",
                  'X6hSL_1': "6h",
                  'X48hSL_3': "48h",
                  'X0h_3': "0h",
                  'X24hSL_2': "24h",
                  'X0h_2': "0h",
                  'X48hS_1': "48h",
                  'X48hSL_1': "48h",
                  'X48hS_2': "48h",
                  'X24hSL_3': "24h",
                  'X6hSL_2': "6h",
                  'X48hSL_2': "48h",
                  'X48hS_3': "48h"
                  }


def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)


def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def omics_stats(X):
    # MORE FILTERS?
    # filter 0: filter out low intensity *******
    # filter 1: remove zero variance features
    # filter 2: remove low CV features to get same order of magnitude for all omics
    # look at dynamic ranges
    # look at ones that are most volatile
    # then look for (same omic level) features that are correlated
    # then look across more omics

    means = X.mean(axis=0)
    sds = X.std(axis=0)
    cvs = np.divide(sds, means)
    high_cv_ids = np.argwhere(np.abs(cvs) > 0.2)  # used to be 0.2
    df_stats = pd.DataFrame(
        {'mean': means,
         'sd': sds,
         'cv': cvs
         })
    return df_stats, high_cv_ids


def scale(df):
    df_scaled = StandardScaler().fit_transform(df.T)
    return pd.DataFrame(df_scaled,
                        columns=df.index,
                        index=df.columns).T


def clean_and_label(d, X, labels):
    # 1. drop patient observations (rows) when > 80% NA (none are filtered
    # out)
    print("before drop na", X.shape)
    X.dropna(axis=0, thresh=np.round(len(X.columns) * 0.80))
    print("after drop na", X.shape)

    # 2. filter out omics measures (columns) that have CV<0.X
    df_stats, high_cv_ids = omics_stats(X)

    # Switch to plotnine?? -- summary stats plots
    # ggplot(aes(x='mean', y='sd'), data=df_stats) + geom_point()
    # ggplot(aes(x='cv'), data=df_stats) + geom_histogram(binwidth=0.2) + \
    #     scale_x_continuous(name="CV", limits=(-3, 5))

    print("before cv filter", X.shape)
    X_high_cv = X.iloc[:, [i[0] for i in high_cv_ids.tolist()]]
    print("after cv filter", X_high_cv.shape)

    # write features/omic names to file
    features = list(X_high_cv)
    outfile = os.path.dirname(os.path.realpath(
        __file__)) + "/model_outputs/" + d + "/feature_names.txt"

    with open(outfile, 'w') as f:
        for feat in features:
            f.write("%s\n" % feat)

    # 3. Get autocorrelation for each patient
    # measure_autocorr() <-- do this for validation of each observation as own
    # entity regardless of patient ID (i.e. each is IID sample)
    # >> measure_autocorr(X_high_cv)

    # can calso do CV for each patient for certian omics? Like what Eric did

    # get labels
    target_names = sorted(list(set(labels)))

    label_map = {}
    values = range(len(target_names))
    for i, key in enumerate(target_names):
        label_map[key] = values[i]

    y = labels.map(label_map).values  # convert to numpy array

    # impute and rescale data -- probably should do each separately!!
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_high_cv)
    # print imputations stats

    # scale full data
    X_scaled = StandardScaler().fit_transform(X_imputed)
    # X_scaled = X_scaled[np.logical_not(np.isnan(X_scaled))]

    X_out = pd.DataFrame(X_scaled)
    X_out.columns = X_high_cv.columns
    X_out.index = X_high_cv.index

    return X_out, y, target_names


def meaure_autocorr():  # will do with time-series addition
    pass


def write_meta(d, time_stamps, labels, ids):

    # # write time stamps
    # outfile = os.path.dirname(os.path.realpath(
    #     __file__)) + "/model_outputs/" + d + "/time.txt"
    # with open(outfile, 'w') as f:
    #     for el in time_stamps:
    #         f.write("%s\n" % el)

    # # write labels
    # outfile = os.path.dirname(os.path.realpath(
    #     __file__)) + "/model_outputs/" + d + "/labels.txt"
    # with open(outfile, 'w') as f:
    #     for el in labels:
    #         f.write("%s\n" % el)

    # # write sample ids
    # outfile = os.path.dirname(os.path.realpath(
    #     __file__)) + "/model_outputs/" + d + "/ids.txt"
    # with open(outfile, 'w') as f:
    #     for el in ids:
    #         f.write("%s\n" % el)

    # write ALL
    outfile = os.path.dirname(os.path.realpath(
        __file__)) + "/model_outputs/" + d + "/all.txt"
    with open(outfile, 'w') as f:
        for i in range(len(ids)):
            f.write("%s,%s,%s\n" % (time_stamps[i], labels[i], ids[i]))


def load_data(d):

    if d == "tcga":
        os.makedirs('data', exist_ok=True)

        if not os.path.isfile(os.path.join('data', 'cnv.csv')):
            urllib.request.urlretrieve(
                url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_cnv.csv',
                filename=os.path.join('data', 'cnv.csv')
            )

        if not os.path.isfile(os.path.join('data', 'gex.csv')):
            urllib.request.urlretrieve(
                url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_gex.csv',
                filename=os.path.join('data', 'gex.csv')
            )

        if not os.path.isfile(os.path.join('data', 'mut.csv')):
            urllib.request.urlretrieve(
                url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_mut.csv',
                filename=os.path.join('data', 'mut.csv')
            )

        if not os.path.isfile(os.path.join('data', 'subtypes.csv')):
            urllib.request.urlretrieve(
                url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_subtypes.csv',
                filename=os.path.join('data', 'subtypes.csv')
            )

        if not os.path.isfile(os.path.join('data', 'survival.csv')):
            urllib.request.urlretrieve(
                url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_survival.csv',
                filename=os.path.join('data', 'survival.csv')
            )

        cnv = pd.read_csv(os.path.join('data', 'cnv.csv'), index_col=0)
        gex = pd.read_csv(os.path.join('data', 'gex.csv'), index_col=0)
        mut = pd.read_csv(os.path.join('data', 'mut.csv'), index_col=0)
        subtypes = pd.read_csv(os.path.join(
            'data', 'subtypes.csv'), index_col=0)
        survival = pd.read_csv(os.path.join(
            'data', 'survival.csv'), index_col=0)

        # use gex + mut + cnv
        gex.index = 'gex_' + gex.index.astype(str)
        mut.index = 'mut_' + mut.index.astype(str)
        cnv.index = 'cnv_' + cnv.index.astype(str)

        # scale separately
        gex_tcga = scale(gex.loc[:, gex.columns.str.contains('TCGA')])
        gex_ccle = scale(gex.loc[:, ~gex.columns.str.contains('TCGA')])

        # merge and scale together
        gex = pd.concat([gex_tcga, gex_ccle], axis=1)
        gex = scale(gex)

        # no need to clean further
        X_out = pd.concat((gex.T, mut.T, cnv.T), axis=1)
        y = subtypes.cms_label
        target_names = sorted(list(set(y)))

        # write files - features/omic names to file
        features = list(X_out)
        outfile = os.path.dirname(os.path.realpath(
            __file__)) + "/model_outputs/" + d + "/feature_names.txt"
        with open(outfile, 'w') as f:
            for feat in features:
                f.write("%s\n" % feat)
        time_stamps = subtypes.loc[:, 'stage']
        sample_ids = subtypes.index

        write_meta(d, time_stamps, y, sample_ids)

        return X_out, y, target_names

    elif d == "motrpac":  # deprecated
        X_rna = pd.read_csv("data/motrpac_rna_only.csv", header=0, index_col=0)
        X_prot = pd.read_csv(
            "data/motrpac_omics_no_rna.csv", header=0, index_col=0)
        labels_r = pd.Series([s[s.find('_') + 1]
                              for s in list(X_rna.index)])
        labels_p = pd.Series([s[s.find('_') + 1]
                              for s in list(X_prot.index)])
        X_out_r, y_r, target_names_r = clean_and_label(d, X_rna, labels_r)
        X_out_p, y_p, target_names_p = clean_and_label(d, X_prot, labels_p)
        return X_out_r, X_out_p, y_r, y_p, target_names_r, target_names_p

    elif d == "ipop":

        X = pd.read_csv("data/ipop_omics.csv", header=0, index_col=0)
        meta = pd.read_csv("data/ipop_meta.csv", header=0, index_col=0)

        time_stamps = meta.loc[:, "rnaseq.CollectionDate"]
        labels = meta.loc[:, "clinic.CL4"]
        sample_ids = X.index
        subject_ids = meta.loc[:, "rnaseq.SubjectID"]

        # scale each omic individualy first before scaled together
        rnaseq = X.loc[:, X.columns.str.startswith(
            'rnaseq')].astype(pd.np.float64)
        prot = X.loc[:, X.columns.str.startswith(
            'prot')].astype(pd.np.float64)
        metab = X.loc[:, X.columns.str.startswith(
            'metab')].astype(pd.np.float64)
        cytok = X.loc[:, X.columns.str.startswith(
            'cytok')].astype(pd.np.float64)
        clinic = X.loc[:, X.columns.str.startswith(
            'clinic')].astype(pd.np.float64)
        # pdb.set_trace()

        rnaseq = scale(rnaseq)
        prot = scale(prot)
        metab = scale(metab)
        cytok = scale(cytok)
        clinic = scale(clinic)

        X = pd.concat((rnaseq, prot, metab, cytok, clinic), axis=1)

        # write meta
        write_meta(d, time_stamps, labels, subject_ids)
        # pdb.set_trace()

        X_out, y, target_names = clean_and_label(d, X, labels)

        # write out filtered data for analysis
        outfile = os.path.dirname(os.path.realpath(
            __file__)) + "/data/" + d + "_X_filtered.csv"
        X_out.to_csv(outfile)

        return X_out, y, target_names

    # elif d == "arabidop1":  # deprecated

    #     # gene = pd.read_csv("data/gene_counts.txt",
    #     #                    sep="\t", header=0, index_col=0)
    #     diff = pd.read_csv("data/gene_fpkm_filtered.txt", sep="\t",
    #                        header=0, index_col=0)  # used to be "differential_expression"
    #     # iso = pd.read_csv("data/isoform_expression.txt", sep="\t",
    #     #                   header=0, index_col=0)
    #     srna = pd.read_csv("data/srna_filtered.txt",
    #                        sep="\t", header=0, index_col=0)  # used to be sRNA_counts
    #     # methyl = pd.read_csv("data/methylation_filtered.txt",
    #     #                      sep="\t", header=0, index_col=0)

    #     # use diff and srna
    #     diff.index = 'diff_' + diff.index.astype(str)
    #     srna.index = 'srna_' + srna.index.astype(str)
    #     X = pd.concat((diff, srna.iloc[:, 1:31]), axis=0).T

    #     ids = X.index
    #     labels = ids.map(label_map1)
    #     time_stamps = ids.map(timestamp_map1)

    #     pdb.set_trace()

    #     # write meta
    #     write_meta(d, time_stamps, labels, ids)

    #     X_out, y, target_names = clean_and_label(d, X, labels)

    #     # MANUAL FILTER FOR SRNA (COUNTS) DATA

    #     # write out filtered data for analysis
    #     outfile = os.path.dirname(os.path.realpath(
    #         __file__)) + "/data/X_" + d + "_filtered.csv"
    #     X_out.to_csv(outfile)

    #     return X_out, y, target_names

    elif d == "arabidop2":

        # gene = pd.read_csv("data/gene_counts.txt",
        #                    sep="\t", header=0, index_col=0)
        diff = pd.read_csv("data/gene_fpkm_filtered.txt", sep="\t",
                           header=0, index_col=0)  # used to be "differential_expression"
        # iso = pd.read_csv("data/isoform_expression.txt", sep="\t",
        #                   header=0, index_col=0)
        srna = pd.read_csv("data/srna_filtered.txt",
                           sep="\t", header=0, index_col=0)  # used to be sRNA_counts
        methyl = pd.read_csv("data/methylation_filtered.txt",
                             sep="\t", header=0, index_col=0)

        # filter for shared samples
        m = methyl.columns
        de = diff.columns
        s = srna.columns

        ids_m = list(m)
        ids_corrected_m = []
        for i in ids_m:
            if i[1] == "_":
                i = i[0] + i[2:]
            ids_corrected_m.append(i)

        ids_cat = list(m) + list(de) + list(s)[1:]
        ids_corrected = []
        for i in ids_cat:
            if i[1] == "_":
                i = i[0] + i[2:]
            ids_corrected.append(i)

        ids_intersect = set(ids_corrected_m).intersection(set(ids_corrected))
        ids = pd.Series(sorted(list(ids_intersect)))
        ids_m = pd.Series(sorted(ids_m))
        ids_de = pd.Series([el[1:] for el in ids])

        # grab correct / shared ids in data
        diff = diff.loc[:, ids_de]  # used to be ids
        srna = srna.loc[:, ids]
        methyl = methyl.loc[:, ids_m]

        # use diff + srna + methyl
        diff.index = 'diff_' + diff.index.astype(str)
        srna.index = 'srna_' + srna.index.astype(str)
        methyl.index = 'methyl_' + methyl.index.astype(str)

        # rename samples and combine
        diff, srna, methyl = diff.T, srna.T, methyl.T
        diff.index = methyl.index
        srna.index = methyl.index

        # scale separately first
        diff = scale(diff)
        srna = scale(srna)
        methyl = scale(methyl)

        X = pd.concat((diff, srna, methyl), axis=1)

        labels = ids.map(label_map2)
        time_stamps = ids.map(timestamp_map2)

        # write meta
        write_meta(d, time_stamps, labels, ids)

        X_out, y, target_names = clean_and_label(d, X, labels)

        # write out filtered data for analysis
        outfile = os.path.dirname(os.path.realpath(
            __file__)) + "/data/X_" + d + "_filtered.csv"
        X_out.to_csv(outfile)

        return X_out, y, target_names

    else:
        print("Error: Pick a valid dataset to run pipeline on!")
        return


def plot_clusters(X, title, vtitle, y, target_names):

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=1., lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.xlabel(vtitle + " 1")
    plt.ylabel(vtitle + " 2")
    plt.show()


def prediction_task(encs, y, method_str, args):

    # run basic classifier
    y_hats, y_probs, y_test = run_logreg(encs, y, args.omic)

    if args.dataset == "ipop":
        # write y probabilities and real ys to file for bootstrapped AUROC
        data = np.append(np.expand_dims(y_test, axis=1), y_probs, axis=1)
        df = pd.DataFrame(
            data, columns=["y_true", "y_pred_1", "y_pred_2", "y_pred_3"])

    elif args.dataset == "motrpac":
        # write y probabilities and real ys to file for bootstrapped AUROC
        data = np.append(np.expand_dims(y_test, axis=1), y_probs, axis=1)
        df = pd.DataFrame(
            data, columns=["y_true", "y_pred_1", "y_pred_2"])

    outfile = os.path.dirname(os.path.realpath(
        __file__)) + "/" + method_str + "_logreg_yhats.txt"
    df.to_csv(outfile, header=True, index=None, sep=',', mode='a')


def map_factors_to_features(z, concatenated_data, pval_threshold=.001):
    """Compute pearson correlation of latent factors with input features.

    Parameters
    ----------
    z:                  (n_samples, n_factors) DataFrame of latent factor values, output of maui model
    concatenated_data:  (n_samples, n_features) DataFrame of concatenated multi-omics data
    Returns
    -------
    feature_s:  DataFrame (n_features, n_latent_factors)
                Latent factors representation of the data X.
    """
    feature_scores = list()
    for j in range(z.shape[1]):
        corrs = pd.DataFrame([stats.pearsonr(concatenated_data.iloc[:, i], z.iloc[:, j]) for i in range(concatenated_data.shape[1])],
                             index=concatenated_data.columns, columns=['r', 'pval'])
        corrs.loc[corrs.pval > pval_threshold, 'r'] = 0
        feature_scores.append(corrs.r)

    feature_s = pd.concat(feature_scores, axis=1)
    feature_s.columns = [i for i in range(z.shape[1])]
    return feature_s
