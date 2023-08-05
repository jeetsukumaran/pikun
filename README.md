## Introduction

`pikun` is a Python package for the analysis and visualization of species delimitation models in an information theoretic framework that provides a true distance or metric space for these models based on the variance of information criterion of [(Meila, 2007)]().

The species delimitation models being analyzed may be generated by any inference package, such as [BP&P](flouri-2018-species-tree), [SNAPP](https://www.beast2.org/snapp/), [DELINEATE](https://github.com/jsukumaran/delineate) etc., or constructed based on taxonomies or classifications based on conceptual descriptions in literature, geography, folk taxonomies, etc.
Regardless of source or basis, each species delimitation model can be considered a *partition* of taxa or lineages and thus can be represented in a dedicated and widely-supported data exchange format, ["`SPART-XML`"](@miralles-2022-spart-versatile), which `pikun` takes as one of its input formats, in addition to DELINEATE.

For every collection of species delimitation models, `pikun` generates a set of partition profiles, partition comparison tables, and a suite of graphical plots visualizing data in these tables.
The partition profiles report unitary information theoretic and other statistics for each of the species delimitation partition, including the probability and entropy of each partition following [@meila-2007-comparing-clusterings].

The partition comparison tables, on the other hand, provide a range of bivariate statistics for every distinct pair of partitions, including the mutual information, joint entropy, etc., as well as a information theoretic distance statistics are true metrics on the space of species distribution models: the variance of information [@meila-2007-comparing-clusterings] and the normalized joint variation of information distance [@vinh-2010-information-theoretic].

## Installation

### Installing from the GitHub Repositories

We recommend that you install directly from the main GitHub repository using pip (which works with an Anaconda environment as well):

```
$ python3 -m pip install --user --upgrade git+https://github.com/jeetsukumaran/pikun.git
```

or

```
$ python3 -m pip install --user --upgrade git+git://github.com/jeetsukumaran/pikun.git
```

## Applications

### Analysis

``pikun-analyze`` is a command-line program that analyzes a collection of partition definitions.

#### Input Formats

``pikun-analyze`` takes as its input a collection of partitions specified in one of the following data formats:

-   A simple list of of lists in JSON format.
    For e.g., given four populations: ``pop1``, ``pop2``, ``pop3``, and ``pop4``:

    ``` json
    [
        [["pop1", "pop2", "pop3", "pop4"]],
        [["pop1"], ["pop2", "pop3", "pop4"]],
        [["pop1", "pop2"], ["pop3", "pop4"]],
        [["pop2"], ["pop1", "pop3", "pop4"]],
        [["pop1"], ["pop2"], ["pop3", "pop4"]],
        [["pop1", "pop2", "pop3"], ["pop4"]],
        [["pop2", "pop3"], ["pop1", "pop4"]],
        [["pop1"], ["pop2", "pop3"], ["pop4"]],
        [["pop1", "pop3"], ["pop2", "pop4"]],
        [["pop3"], ["pop1", "pop2", "pop4"]],
        [["pop1"], ["pop3"], ["pop2", "pop4"]],
        [["pop1", "pop2"], ["pop3"], ["pop4"]],
        [["pop2"], ["pop1", "pop3"], ["pop4"]],
        [["pop2"], ["pop3"], ["pop1", "pop4"]],
        [["pop1"], ["pop2"], ["pop3"], ["pop4"]]
    ]
    ```

    This can be explicitly specified by passing the argument "json-list" to the ``-f`` or ``--format`` option:

    ```
    $ pikun-analyze -f json-list partitions.json
    $ pikun-analyze --format json-list partitions.json
    ```

-   [DELINEATE](https://github.com/jsukumaran/delineate)

    ```
    $ pikun-analyze -f delineate delineate-results.json
    $ pikun-analyze --format delineate delineate-results.json
    ```

- SPART-XML

    ```
    $ pikun-analyze -f spart-xml data.xml
    $ pikun-analyze --format spart-xml data.xml
    ```

#### Analysis Options

-   The output file names and paths can be specified by using the ``-o``/``--output-title`` and ``-O``/``--output-directory``

    ```
    $ pikun-analyze \
        -f delineate \
        -o project42 \
        -O analysis_dir \
        delineate-results.json
    $ pikun-analyze \
        --format delineate \
        --output-title project42 \
        --output-directory analysis_dir \
        delineate-results.json
    ```

-   The number of partitions can are read from the input set can be restricted to the first $n$ partitions using the ``--limit-partitions`` option:

    ```
    $ pikun-analyze \
        --format delineate \
        --output-title project42 \
        --output-directory analysis_dir \
        --limit 10 \
        delineate-results.json
    ```

    This is option is particularly useful when the number of partitions in the input is large and/or most of the partitions in the input set may not be of interest.
    For e.g., a typical [DELINEATE](https://github.com/jsukumaran/delineate) analysis may generate hundreds if not thousands of partitions, and most of these are low-probability ones of not much practical interest.
    Using the ``--limit`` flag will focus on just the subset of interest, which will help with computation time and resources.

#### Output

``pikun-analyze`` will generate two tab-delimited (``.tsv``) files (named and located based on the ``-o``/``--output-title`` and ``-O``/``--output-directory`` options):

- ``output-directory/output-title-profiles.tsv``
- ``output-directory/output-title-comparisons.tsv``

These files provide univariate and a mix of univariate and bivariate statistics, respectively, for the partitions.

Both of these files can be directly loaded as a PANDAS data frame for more detailed analysis:

```
>>> import pandas as pd
>>> df1 = pd.read_cs(
...     "output-directory/output-title-comparisons.tsv",
...     delimiter="\t"
... )
```

The ``-comparisons`` file includes the variance of information distance statistics: ``vi_distance`` and ``vi_normalized_kraskov``.














