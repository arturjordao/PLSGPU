import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_classification
from pls_gpu import PLSGPU
import time


if __name__ == '__main__':
    np.random.seed(12227)

    X, y = make_classification(n_samples=10000, n_features=3000, n_classes=2, n_clusters_per_class=1)

    pls = PLSRegression(n_components=10)
    pls.fit(X, y)
    start = time.time()
    pls.transform(X)
    end = time.time()
    print('Projection time PLS [{:.4f}]'.format(end-start))

    pls_gpu = PLSGPU(pls, X.shape[0])
    start = time.time()
    pls_gpu.transform(X)
    end = time.time()
    print('Projection time PLSGPU [{:.4f}]'.format(end - start))