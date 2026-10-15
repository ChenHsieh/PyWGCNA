"""TOMsimilarity must give exactly the same values as the original full-matrix formula."""
import numpy as np
import pytest

from PyWGCNA.wgcna import WGCNA


def reference_tom(adjMat, TOMDenom, TOMType):
    """The full-matrix implementation used before the row-block version."""
    adjMat = adjMat.copy()
    np.fill_diagonal(adjMat, 0)
    L = adjMat @ adjMat
    ki = adjMat.sum(axis=1)
    kj = adjMat.sum(axis=0)
    if TOMDenom == 0:
        MINK = np.array([np.minimum(ki_, kj) for ki_ in ki])
    else:
        MINK = np.array([((ki_ + kj) / 2) for ki_ in ki])
    if TOMType == 0:
        tom = (L + adjMat) / (MINK + 1 - adjMat)
    else:
        tom = np.fabs((L + adjMat)) / (MINK + 1 - np.fabs(adjMat))
    np.fill_diagonal(tom, 1)
    return tom


def random_adjacency(n, signed, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(30, n))
    cor = np.corrcoef(x, rowvar=False)
    return cor ** 3 if signed else np.abs(cor) ** 6


# 5000 genes spans more than two row blocks of 2048
@pytest.mark.parametrize("n", [10, 2048, 2049, 5000])
@pytest.mark.parametrize("TOMType", [0, 1])
@pytest.mark.parametrize("TOMDenom", [0, 1])
def test_tom_matches_full_matrix_formula(n, TOMType, TOMDenom):
    adj = random_adjacency(n, signed=TOMType == 1, seed=n)
    expected = reference_tom(adj, TOMDenom, TOMType)
    got = WGCNA.TomSimilarityFromAdj(adj.copy(), TOMDenom, TOMType)
    assert np.array_equal(got, expected)


def test_asymmetric_adjacency_is_still_rejected():
    adj = random_adjacency(3000, signed=False)
    adj[2500, 10] += 0.1
    with pytest.raises(SystemExit):
        WGCNA.checkAdjMat(adj)
