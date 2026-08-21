"""kmer-dust: alignment-free clustering of genomic bins.

Tile assemblies into fixed-width bins, sketch each bin with a FracMinHash of its
canonical k-mers, assemble a sparse bin x k-mer matrix over many haplotypes,
factor it with a randomized SVD, embed with UMAP and cluster with HDBSCAN --
then paint the clusters back onto T2T-CHM13 annotation tracks and onto every
contributing HPRC assembly.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
