#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
## Copyright (c) 2023 Jeet Sukumaran.
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
##     * Redistributions of source code must retain the above copyright
##       notice, this list of conditions and the following disclaimer.
##     * Redistributions in binary form must reproduce the above copyright
##       notice, this list of conditions and the following disclaimer in the
##       documentation and/or other materials provided with the distribution.
##     * The names of its contributors may not be used to endorse or promote
##       products derived from this software without specific prior written
##       permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
## ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
## WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL JEET SUKUMARAN BE LIABLE FOR ANY DIRECT,
## INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
## BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
## OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##
##############################################################################

import os
import pathlib
import sys
import argparse
import json
import math
import functools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay

import yakherd
from pikun import partitionmodel
from pikun import utility

def mirror_dataframe(df, col1, col2, col3):
    """
    Create a mirrored dataframe based on the input columns.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    col1 (str): The first column to be mirrored.
    col2 (str): The second column to be mirrored.
    col3 (str): The third column to be included.

    Returns:
    pd.DataFrame: The resulting mirrored dataframe.
    """

    df_1 = df[[col1, col2, col3]].copy()
    df_2 = df_1.copy()
    df_2[[col1, col2]] = df_2[[col2, col1]]

    # Append df_mirror to the relevant columns of df
    df_final = pd.concat([df_1, df_2], ignore_index=True)
    return df_final

def contour_plot(
    X,
    Y,
    Z,
    num_points=100,
    xscale='linear',
    yscale='linear',
    colormap='coolwarm',
    overlay_hexbin=False
):
    xi = np.linspace(X.min(), X.max(), num_points)
    yi = np.linspace(Y.min(), Y.max(), num_points)
    zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='linear')

    # Generate a mask for the interpolated grid
    tri = Delaunay(np.vstack((X, Y)).T)
    X_grid, Y_grid = np.meshgrid(xi, yi)
    mask = tri.find_simplex(np.vstack((X_grid.flatten(), Y_grid.flatten())).T) < 0
    mask = mask.reshape(X_grid.shape)
    zi_masked = np.ma.array(zi, mask=mask)

    plt.figure()
    plt.contour(xi, yi, zi_masked, levels=14, linewidths=0.5, colors='k')
    cntr = plt.contourf(xi, yi, zi_masked, levels=14, cmap=colormap)
    # Overlay hexbin plot if overlay_hexbin is True
    if overlay_hexbin:
        plt.hexbin(X, Y, C=Z, gridsize=50, cmap=colormap, reduce_C_function=np.mean, mincnt=1, xscale=xscale, yscale=yscale)
    plt.xscale(xscale)
    plt.yscale(yscale)

class Plotter(utility.RuntimeClient):

    def __init__(
        self,
        config_d=None,
        runtime_context=None,
    ):
        super().__init__(runtime_context=runtime_context)
        self.configure(config_d=config_d)

    def configure(self, config_d):
        pass

    def prep_df_for_heatmap(
        self,
        df,
        value_fieldname,
        is_index_first_partition=False,
    ):
        # print(df)
        if is_index_first_partition:
            c_df = df.pivot(
                index="ptn1",
                columns="ptn2",
                values=value_fieldname,
            )
            c_df = utility.mirror_upper_half(c_df)
        else:
            c_df = df.pivot(
                index="ptn2",
                columns="ptn1",
                values=value_fieldname,
            )
            c_df = utility.mirror_lower_half(c_df)
        np.fill_diagonal(c_df.values, 0.0)
        c_df = c_df.round(12)
        return c_df

    def plot(self, df):
        plot_format = "pdf"
        self.plot_clustermaps(
            df=df,
            plot_format=plot_format
        )
        self.plot_support_vs_distance(
            df=df,
            plot_format=plot_format,
        )

    def plot_support_vs_distance(
        self,
        df,
        plot_format="pdf",
    ):

        pf1_key = "ptn1_support"
        pf2_key = "ptn2_support"
        dist_key = "vi_normalized_kraskov"
        df = df[df["ptn1"] != df["ptn2"]]
        c_df = mirror_dataframe(df, pf1_key, pf2_key, dist_key)
        x = c_df[pf1_key]
        y = c_df[pf2_key]
        z = c_df[dist_key]
        contour_plot(x, y, z, xscale="log", yscale="log")
        store = self.runtime_context.ensure_store(
            key="support",
            name_parts=["vi_distance_vs_support"],
            extension=plot_format,
            )
        self.finish_plot(
            output_path=store.path,
            plot_format="pdf",
            )

    def plot_clustermaps(
        self,
        df,
        plot_format="pdf"
    ):
        for value_fieldname in (
            "vi_distance",
            "vi_normalized_kraskov",
            ):
            for is_cluster in (True, False):
                self.plot_clustermap(
                    df,
                    value_fieldname=value_fieldname,
                    is_cluster_rows=is_cluster,
                    is_cluster_cols=is_cluster,
                )
                if is_cluster:
                    subtype = "reordered"
                else:
                    subtype = "unordered"
                name_parts = [
                    value_fieldname,
                    subtype,
                ]
                store = self.runtime_context.ensure_store(
                    key="+".join(name_parts),
                    name_parts=name_parts,
                    extension=plot_format,
                )
                self.finish_plot(
                    output_path=store.path,
                    plot_format="pdf",
                )

    def plot_clustermap(
        self,
        df,
        value_fieldname,
        is_store_data=True,
        is_store_plot=True,
        is_show_plot=False,
        is_index_first_partition=False,
        is_cluster_rows=False,
        is_cluster_cols=False,
        plot_format="pdf",
        plot_kwargs=None,
    ):
        c_df = self.prep_df_for_heatmap(
            df=df,
            value_fieldname=value_fieldname,
            is_index_first_partition=is_index_first_partition,
        )
        if not plot_kwargs:
            plot_kwargs = {}
        try:
            # check to see if it can be compressed
            dist_array = squareform(c_df)
            dist_linkage = hierarchy.linkage(dist_array)
            plot_kwargs["row_linkage"] = dist_linkage
            plot_kwargs["col_linkage"] = dist_linkage
        except ValueError:
            pass

        plot_kwargs["row_cluster"] = is_cluster_rows
        plot_kwargs["col_cluster"] = is_cluster_cols

        g = sns.clustermap(
            c_df,
            **plot_kwargs,
        )

    def finish_plot(
        self,
        output_path,
        plot_format="pdf",
        is_store_plot=True,
        is_show_plot=False,
    ):
        if is_store_plot:
            plt.savefig(
                output_path,
                format=plot_format,
                )
        if is_show_plot:
            plt.show()
        plt.close()


def main(args=None):
    parent_parser = argparse.ArgumentParser()
    parent_parser.set_defaults(func=lambda x: parent_parser.print_help())
    input_options = parent_parser.add_argument_group("Input Options")
    input_options.add_argument(
        "src_path",
        action="store",
        metavar="FILE",
        nargs="+",
        help="Path to data source file.",
    )
    output_options = parent_parser.add_argument_group("Output Options")
    output_options.add_argument(
        "-o",
        "--output-title",
        action="store",
        default="pikun",
        help="Prefix for output filenames [default='%(default)s'].",
    )
    output_options.add_argument(
        "-O",
        "--output-directory",
        action="store",
        default=os.curdir,
        help="Directory for output files [default='%(default)s'].",
    )
    # cluster_plot_options = parent_parser.add_argument_group("Cluster Plot Options")
    # cluster_plot_options.add_argument(
    #     "--cluster-rows",
    #     action=argparse.BooleanOptionalAction,
    #     dest="is_cluster_rows",
    #     default=False,
    #     help="Reorder / do not reorder partition rows to show clusters clearly",
    # )
    # cluster_plot_options.add_argument(
    #     "--cluster-cols",
    #     action=argparse.BooleanOptionalAction,
    #     dest="is_cluster_cols",
    #     default=False,
    #     help="Reorder / do not reorder partition colums to show clusters clearly",
    # )

    logger_configuration_parser = yakherd.LoggerConfigurationParser(name="pikun")
    logger_configuration_parser.attach(parent_parser)
    logger_configuration_parser.console_logging_parser_group.add_argument(
        "--progress-report-frequency",
        type=int,
        action="store",
        help="Frequency of progress reporting.",
    )
    args = parent_parser.parse_args(args)
    config_d = dict(vars(args))
    logger = logger_configuration_parser.get_logger(args_d=config_d)
    runtime_context = utility.RuntimeContext(
        logger=logger,
        random_seed=None,
        output_directory=args.output_directory,
        output_title=args.output_title,
        output_configuration=config_d,
    )

    plotter = Plotter(
        runtime_context=runtime_context,
        config_d=config_d,
    )
    df = utility.read_files_to_dataframe(filepaths=args.src_path)
    plotter.plot(df)


if __name__ == "__main__":
    main()

