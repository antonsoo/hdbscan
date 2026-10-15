import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from hdbscan import robust_single_linkage


@pytest.mark.parametrize('alpha', [None, 1., 2.])
@pytest.mark.parametrize('mode', ['dense', 'sparse', 'precomputed', 'callable'])
def test_rsl_generic_alpha(alpha, mode):
    data = np.array([[0.], [1.], [4.], [5.], [10.]])
    distances = squareform(pdist(data))
    effective_alpha = np.sqrt(2.) if alpha is None else alpha
    core = np.sort(distances, axis=0)[1]
    reachability = np.maximum(
        np.maximum(core[:, None], core[None, :]), distances / effective_alpha,
    )
    np.fill_diagonal(reachability, 0.)
    expected = linkage(squareform(reachability), method='single')

    metric = 'euclidean'
    if mode == 'sparse':
        data = sparse.csr_matrix(data)
    elif mode == 'precomputed':
        data = distances.copy()
        metric = 'precomputed'
    elif mode == 'callable':
        metric = lambda x, y: np.linalg.norm(x - y)
    kwargs = {} if alpha is None else {'alpha': alpha}
    labels, tree = robust_single_linkage(
        data, cut=2., k=1, gamma=1, algorithm='generic', metric=metric, **kwargs,
    )
    assert_allclose(tree[:, 2], expected[:, 2])
    # At alpha=2 the two pairs join; at alpha=1 or sqrt(2) they stay separate.
    expected_labels = np.array([0, 0, 0, 0, 1] if alpha == 2. else [0, 0, 1, 1, 2])
    assert_array_equal(
        labels[:, None] == labels, expected_labels[:, None] == expected_labels,
    )
