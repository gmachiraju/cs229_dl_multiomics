from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import *
from utils import *

import pandas as pd
import numpy as np
import scipy as sci

from itertools import chain

import pdb
import os
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_ungrouped_models(X, y, target_names, e_size, args):

    # fit / train the models on single omic -> rna seq
    pca_loadings, pca_transformed = run_pca(X.values, e_size)
    fa_loadings, fa_transformed = run_fa(X.values, e_size)
    # tsne_loadings, tsne_transformed = run_tsne(X.values, e_size) #<--- e_size > 4 is an issue
    # umap_transformed = run_umap(rna.values, e_size)
    ael_loadings, ael_transformed = run_vanilla_autoencoder(
        X.values, "linear", e_size)
    aer_loadings, aer_transformed = run_vanilla_autoencoder(
        X.values, "relu", e_size)
    # aes_loadings, aes_transformed = run_basic_autoencoder(
    #     X.values, "sigmoid", e_size)

    # transpose AE models for consistency
    ael_loadings = ael_loadings.T
    aer_loadings = aer_loadings.T

    method_strings = ["pca", "fa", "ael", "aer"]
    loadings = [pca_loadings, fa_loadings, ael_loadings, aer_loadings]
    transformed = [pca_transformed, fa_transformed,
                   ael_transformed, aer_transformed]

    # write to df and to csv
    print("\nloading dims:\n---------------------------")
    for i, l in enumerate(loadings):
        print(method_strings[i] + ": ", l.shape)
        df = pd.DataFrame.from_records(l)
        outfile = os.path.dirname(os.path.realpath(
            __file__)) + "/model_outputs/" + args.dataset + "/loadings_" + args.num_factors + "/" + method_strings[i] + "_loadings.txt"
        df.to_csv(outfile, header=True, index=None, sep=',', mode='a')

    print("\ntransformed dims:\n---------------------------")
    for i, l in enumerate(transformed):
        print(method_strings[i] + ": ", l.shape)
        df = pd.DataFrame.from_records(l)
        outfile = os.path.dirname(os.path.realpath(
            __file__)) + "/model_outputs/" + args.dataset + "/transformed_" + args.num_factors + "/" + method_strings[i] + "_transformed.txt"
        df.to_csv(outfile, header=True, index=None, sep=',', mode='a')

    return transformed, method_strings


def run_grouped_models(X, y, target_names, e_size, args):

    # added hyperparam for maui
    h = X.shape[1]

    mofa_loadings, mofa_transformed = run_mofa(
        X, args, e_size)
    maui_loadings, maui_transformed = run_maui(X, args, e_size, h, epochs=200)
    # oivae_encodings = run_oivae(X, args, e_size)

    # get comparable shapes
    maui_loadings = maui_loadings.T
    mofa_transformed = mofa_transformed.T

    method_strings = ["mfa", "maui"]
    loadings = [mofa_loadings, maui_loadings]
    transformed = [mofa_transformed, maui_transformed]

    # write to df and to csv
    print("\nloading dims:\n---------------------------")
    for i, l in enumerate(loadings):
        print(method_strings[i] + ": ", l.shape)
        df = pd.DataFrame.from_records(l)
        outfile = os.path.dirname(os.path.realpath(__file__)) + "/model_outputs/" + args.dataset + \
            "/loadings_" + args.num_factors + "/" + \
            method_strings[i] + "_loadings.txt"
        df.to_csv(outfile, header=True, index=None, sep=',', mode='a')

    print("\ntransformed dims:\n---------------------------")
    for i, l in enumerate(transformed):
        print(method_strings[i] + ": ", l.shape)
        df = pd.DataFrame.from_records(l)
        outfile = os.path.dirname(os.path.realpath(__file__)) + "/model_outputs/" + args.dataset + \
            "/transformed_" + args.num_factors + "/" + \
            method_strings[i] + "_transformed.txt"
        df.to_csv(outfile, header=True, index=None, sep=',', mode='a')

    return transformed, method_strings


def main():

    # parser for type of analysis
    parser = argparse.ArgumentParser(description='Process multiple omics.')
    parser.add_argument(
        '-m', '--mode', help='Mode? Run or analysis.', required=True)
    parser.add_argument(
        '-d', '--dataset', help='MoTrPAC, iPOP, Arabidop1/2, TCGA data to run on?', required=True)
    parser.add_argument(
        '-o', '--omic', help='Type of omics analysis: single or multi', required=True)
    parser.add_argument(
        '-n', '--num_factors', help='Number of latent components/factors', required=False, default=10)
    parser.add_argument(
        '-cmfa', '--cached_mfa', type=str2bool, nargs='?', help='Cached MFA model? Does one exist and would you like to use that run?', required=False, default=False, const=True)
    parser.add_argument(
        '-cvae', '--cached_vae', type=str2bool, nargs='?', help='Cached VAE model? Does one exist and would you like to use that run?', required=False, default=False, const=True)
    parser.add_argument(
        '-v', '--verbosity', type=str2bool, nargs='?', help='Generate component printouts and plots?', required=False, default=False, const=True)
    args = parser.parse_args()

    # read the data
    if args.dataset == "ipop":
        X, y, target_names = load_data(args.dataset)
        Xr, yr, target_names_r = X.loc[
            :, X.columns.str.startswith('rnaseq')], y, target_names
    elif args.dataset == "motrpac":
        Xr, X, yr, y, target_names_r, target_names = load_data(
            args.dataset)
    elif args.dataset == "arabidop1" or args.dataset == "arabidop2":
        X, y, target_names = load_data(args.dataset)
        # not supporting single omics here
    elif args.dataset == "tcga":
        X, y, target_names = load_data(args.dataset)
        # not supporting single omics here

    e_size = int(args.num_factors)  # encoding size / number of components (L)

    if args.mode == "run":
        if args.omic == 'single':

            try:
                encs_by_method, method_strings = run_ungrouped_models(
                    Xr, yr, target_names_r, e_size, args.verbosity)
            except:
                print("no single-omics data detected")
                return

            # for i, encs in enumerate(encs_by_method):
            # prediction_task(encs, yr, method_strings[i], args)

        # fit / train the models on multiple (all 5) omics
        elif args.omic == 'multi':

            # Naive approach with stacked data! (ungrouped)
            #-----------------------------------
            encs_by_method_u, method_strings_u = run_ungrouped_models(
                X, y, target_names, e_size, args)

            # for i, encs in enumerate(encs_by_method_u):
            #     prediction_task(encs, y, method_strings_u[i], args)

            # Grouped analysis!
            #-------------------
            encs_by_method_g, method_strings_g = run_grouped_models(
                X, y, target_names, e_size, args)
            # for i, encs in enumerate(encs_by_method_g):
            #     prediction_task(encs, y, method_strings_g[i], args)

    elif args.mode == "analysis":
        print("run analysis script instead.")
        pass

    elif args.mode == "maui_only":

        args.num_factors = str(70)
        e_size = 70
        h = X.shape[1]

        loadings, transformed = run_maui(
            X, args, encoding_dim=e_size, hidden_dim=h)  # used to be default h = 1100
        print(loadings.shape)
        print(transformed.shape)

        # write to df and to csv
        print("\nloading dims:\n---------------------------")
        print("maui" + ": ", loadings.shape)
        df = pd.DataFrame.from_records(loadings)
        outfile = os.path.dirname(os.path.realpath(__file__)) + "/model_outputs/" + args.dataset + \
            "/loadings_" + args.num_factors + "/" + "maui" + "_loadings.txt"
        df.to_csv(outfile, header=True, index=None, sep=',', mode='a')

        print("\ntransformed dims:\n---------------------------")
        print("maui" + ": ", transformed.shape)
        df = pd.DataFrame.from_records(transformed)
        outfile = os.path.dirname(os.path.realpath(__file__)) + "/model_outputs/" + args.dataset + \
            "/transformed_" + args.num_factors + "/" + "maui" + "_transformed.txt"
        df.to_csv(outfile, header=True, index=None, sep=',', mode='a')

        pdb.set_trace()

if __name__ == "__main__":
    main()
