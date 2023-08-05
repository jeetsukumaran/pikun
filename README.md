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


## Input Formats

`pikun` currently supports the following data formats:

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

-   [DELINEATE](https://github.com/jsukumaran/delineate)

- SPART-XML

## Applications

## Workflow



