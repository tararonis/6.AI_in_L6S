from typing import Type, Union, Dict, Any

from sklearn.neighbors import LocalOutlierFactor

from sberpm._holder import DataHolder
from sberpm.ml.anomaly_detection._outlier_detector import OutlierDetector


class OutlierLOF(OutlierDetector):
    """
    Detects outliers using Local Outlier Factor and gives
        basic information about them.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names
            of its necessary columns.

    n_neighbors : int, default=20
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        metric used for the distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics:
        https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    p : int, default=2
        Parameter for the Minkowski metric from
        :func:`sklearn.metrics.pairwise.pairwise_distances`. When p = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the scores of the samples.

        - if 'auto', the threshold is determined as in the
          original paper,
        - if a float, the contamination should be in the range [0, 0.5].

    novelty : bool, default=False
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        that you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names
        of its necessary columns.

    _estimator: object
        Algorithm that will be used to detect outliers. It must have either
        both "fit" and "predict" methods or "fit_predict" method.

    _outlier_label : int
        A label that the estimator will use to mark outliers.

    Examples
    --------
    >>> from sberpm import DataHolder
    >>> from sberpm.ml.anomaly_detection import OutlierLOF
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ... 'id_column': [1, 1, 2],
    ... 'activity_column':['st1', 'st2', 'st1'],
    ... 'dt_column':[123456, 123457, 123458]})
    >>> dh = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> outlier_detector = OutlierLOF(dh).
    ...     add_groupby_feature('max_time', dh.duration_column, max)
    >>> outlier_detector.apply()
    >>>
    >>> outlier_detector.get_outlier_ids()
    >>> outlier_detector.print_result()
    >>> outlier_detector.show_permutation_importance()
    """

    def __init__(
        self,
        data_holder: Type[DataHolder],
        n_neighbors: int = 20,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params: Dict[str, Any] = None,
        contamination: Union[str, float] = "auto",
        n_jobs: int = None,
    ):
        super(OutlierLOF, self).__init__(data_holder)
        self._estimator = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            contamination=contamination,
            novelty=True,
            n_jobs=n_jobs,
        )
        self._outlier_label = -1
