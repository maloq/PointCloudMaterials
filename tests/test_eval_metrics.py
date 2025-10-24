import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sys,os
sys.path.append(os.getcwd())
from src.eval_pipeline.metrics.silhouette import SilhouetteMetric
from src.eval_pipeline.metrics.ari import ARIMetric


def _make_blobs():
    rng = np.random.default_rng(0)
    return np.vstack([
        rng.normal(loc=0.0, scale=0.1, size=(10, 2)),
        rng.normal(loc=1.0, scale=0.1, size=(10, 2)),
    ])


def test_silhouette_metric():
    X = _make_blobs()
    kmeans = KMeans(n_clusters=2, n_init="auto")
    labels = kmeans.fit_predict(X)
    metric = SilhouetteMetric()
    score = metric.run_once(X, cluster_labels=labels)
    assert 0 <= score <= 1


def test_ari_metric():
    X = _make_blobs()
    true_labels = np.array([0] * 10 + [1] * 10)
    kmeans = KMeans(n_clusters=2, n_init="auto")
    cluster_labels = kmeans.fit_predict(X)
    metric = ARIMetric()
    score = metric.run_once(X, true_labels=true_labels, cluster_labels=cluster_labels)
    assert score > 0.5


def test_gmm_clustering():
    X = _make_blobs()
    gmm = GaussianMixture(n_components=2)
    labels = gmm.fit_predict(X)
    assert set(labels) == {0, 1}
