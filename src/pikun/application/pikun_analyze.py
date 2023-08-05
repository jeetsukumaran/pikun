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
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from matplotlib.colors import LogNorm, Normalize

import yakherd
from pikun import partitionmodel
from pikun import utility


class PartitionCoordinator(utility.RuntimeClient):

    def __init__(
        self,
        config_d=None,
        log_base=2,
        runtime_context=None,
    ):
        super().__init__(runtime_context=runtime_context)
        self.log_base = log_base
        self.configure(config_d=config_d)
        self.reset()

    def configure(self, config_d):
        if not config_d:
            return
        self.log_frequency = config_d.get("progress_report_frequency", 0.01)
        if self.log_frequency is None:
            self.log_frequency = 0.01

    def reset(self):
        self.partitions = []

    def new_partition(self, **kwargs):
        kwargs["log_base"] = self.log_base
        ptn = partitionmodel.Partition(**kwargs)
        self.partitions.append(ptn)
        return ptn

    def read_partitions(
        self,
        src_paths,
        data_format,
        limit_records=None,
    ):
        yfn = None
        if data_format == "delineate":
            yfn = self.parse_delineate
        elif data_format == "json-list":
            yfn = self.parse_json
        elif data_format == "spart-xml":
            yfn = self.parse_spart_xml
        else:
            raise ValueError(data_format)
        n_partitions_parsed = 0
        for src_idx, src_path in enumerate(src_paths):
            self.logger.log_info(f"Reading source {src_idx+1} of {len(src_paths)}: '{src_path}'")
            with open(src_path) as src:
                src_data = src.read()
            for pidx, partition in enumerate(yfn(src_data)):
                n_partitions_parsed += 1
                if limit_records and n_partitions_parsed == limit_records:
                    break

    def parse_delineate(self, src_data):
        delineate_results = json.loads(src_data)
        src_partitions = delineate_results["partitions"]
        self.logger.log_info(f"{len(src_partitions)} partitions in source")
        for spart_idx, src_partition in enumerate(src_partitions):
            self.logger.log_info(f"Storing partition {spart_idx+1} of {len(src_partitions)}")
            metadata_d = {
                "p_dc": src_partition.get("constrained_probability", 0),
                "p_du": src_partition.get("unconstrained_probability", 0),
                "support": src_partition.get("unconstrained_probability", 0),
            }
            kwargs = {
                "label": spart_idx + 1,
                "metadata_d": metadata_d,
            }
            partition_data = src_partition["species_leafsets"]
            if not isinstance(partition_data, dict):
                # legacy format!
                kwargs["subsets"] = partition_data
            else:
                kwargs["partition_d"] = partition_data
            partition = self.new_partition(**kwargs)
            yield partition

    def parse_json(self, src_data):
        src_data = json.loads(src_data)
        for ptn_idx, ptn in enumerate(src_data):
            partition = self.new_partition(
                label=ptn_idx + 1,
                subsets=ptn,
            )
            yield partition

    def parse_spart_xml(self, src_data):
        root = ET.fromstring(src_data)
        for spart_idx, spartition_element in enumerate(root.findall(".//spartition")):
            subsets = []
            for subset_idx, subset_element in enumerate(
                spartition_element.findall(".//subset")
            ):
                subset = []
                for individual_element in subset_element.findall(".//individual"):
                    subset.append(individual_element.get("ref"))
                subsets.append(subset)
            partition = self.new_partition(
                label=spartition_element.attrib.get("label", spart_idx + 1),
                subsets=subsets,
            )
            yield partition

    @property
    def partition_list(self):
        if not hasattr(self, "_partition_list") or self._partition_list is None:
            self._partition_list = list(self.partitions.values())
        return self._partition_list

    @property
    def partition_profile_store(self):
        if (
            not hasattr(self, "_partition_profile_store")
            or self._partition_profile_store is None
        ):
            self._partition_profile_store = self.runtime_context.ensure_store(
                key="partition-profile",
                name_parts=[
                    "partition",
                    "profiles",
                ],
                separator="-",
                extension="tsv",
            )
        return self._partition_profile_store

    @property
    def partition_comparison_store(self):
        if (
            not hasattr(self, "_partition_comparison_store")
            or self._partition_comparison_store is None
        ):
            self._partition_comparison_store = self.runtime_context.ensure_store(
                key="partition-comparison",
                name_parts=["partition", "comparisons"],
                separator="-",
                extension="tsv",
            )
        return self._partition_comparison_store

    def analyze_partitions(self, is_mirror=False):
        if is_mirror:
            n_expected_cmps = len(self.partitions) * len(self.partitions)
        else:
            n_expected_cmps = int(len(self.partitions) * len(self.partitions) / 2)
        progress_step = int(n_expected_cmps * self.log_frequency)
        if progress_step < 1:
            progress_step = 1
        n_comparisons = 0
        comparisons = []
        seen_compares = set()
        for pidx1, ptn1 in enumerate(self.partitions):
            profile_d = {
                "partition_id": pidx1,
                "label": ptn1.label,
                "n_elements": ptn1.n_elements,
                "n_subsets": ptn1.n_subsets,
                "vi_entropy": ptn1.vi_entropy(),
            }
            if ptn1.metadata_d:
                profile_d.update(ptn1.metadata_d)
            self.partition_profile_store.write_d(profile_d)
            ptn1_metadata = {}
            for k, v in ptn1.metadata_d.items():
                ptn1_metadata[f"ptn1_{k}"] = v
            for pidx2, ptn2 in enumerate(self.partitions):
                cmp_key = frozenset([ptn1, ptn2])
                if not is_mirror and cmp_key in seen_compares:
                    continue
                seen_compares.add(cmp_key)
                if n_comparisons == 0 or (n_comparisons % progress_step) == 0:
                    self.logger.log_info(f"[ {int(n_comparisons * 100/n_expected_cmps): 4d} % ] Comparison {n_comparisons} of {n_expected_cmps}: Partition {ptn1.label} vs. partition {ptn2.label}")
                n_comparisons += 1
                comparison_d = {
                    "ptn1": ptn1.label,
                    "ptn2": ptn2.label,
                }
                comparison_d.update(ptn1_metadata)
                for k, v in ptn2.metadata_d.items():
                    comparison_d[f"ptn2_{k}"] = v
                comparison_d["vi_entropy_ptn1"] = ptn1.vi_entropy()
                comparison_d["vi_entropy_ptn2"] = ptn2.vi_entropy()
                for value_fieldname, value_fn in (
                    ("vi_mi", ptn1.vi_mutual_information),
                    ("vi_joint_entropy", ptn1.vi_joint_entropy),
                    ("vi_distance", ptn1.vi_distance),
                    ("vi_normalized_kraskov", ptn1.vi_normalized_kraskov),
                ):
                    comparison_d[value_fieldname] = value_fn(ptn2)
                self.partition_comparison_store.write_d(comparison_d)
                comparisons.append(comparison_d)
        df = pd.DataFrame.from_records(comparisons)
        return df

def main(args=None):
    parent_parser = argparse.ArgumentParser()
    parent_parser.set_defaults(func=lambda x: parent_parser.print_help())
    # subparsers = parent_parser.add_subparsers()

    # analyze_parser = subparsers.add_parser("analyze", help="Analyze a collection of partitions.")
    # parent_parser.set_defaults(func=execute_analysis)
    input_options = parent_parser.add_argument_group("Input Options")
    input_options.add_argument(
        "src_path",
        action="store",
        metavar="FILE",
        nargs="+",
        help="Path to data source file.",
    )
    input_options.add_argument(
        "-f",
        "--format",
        action="store",
        default="delineate",
        help="Format for partition data [default=%(default)s].",
    )
    input_options.add_argument(
        "--limit-partitions",
        action="store",
        default=None,
        type=int,
        help="Limit data to this number of partitions.",
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
    pc = PartitionCoordinator(
        config_d=config_d,
        runtime_context=runtime_context,
    )
    pc.read_partitions(
        src_paths=config_d["src_path"],
        data_format=config_d["format"],
        limit_records=config_d["limit_partitions"],
    )
    df = pc.analyze_partitions()

if __name__ == "__main__":
    main()
