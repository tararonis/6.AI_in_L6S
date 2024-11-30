from __future__ import annotations

from typing import Union, TYPE_CHECKING, Optional, Callable

from sberpm.ml.anomaly_detection._outlier_detector import OutlierDetector
from sberpm.ml.anomaly_detection._outlier_detector import PyodModuleNotFoundError


class OutlierCBLOF(OutlierDetector):
    """
    Detects outliers using Cluster-Based Local Outlier Factor
    and gives basic information about them.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the
            names of its necessary columns.

    n_clusters : int, optional (default=8)
        The number of clusters to form as well as the number of
        centroids to generate.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    clustering_estimator : Estimator, optional (default=None)
        The base clustering algorithm for performing data clustering.
        A valid clustering algorithm should be passed in. The estimator should
        have standard sklearn APIs, fit() and predict(). The estimator should
        have attributes ``labels_`` and ``cluster_centers_``.
        If ``cluster_centers_`` is not in the attributes once the model is fit,
        it is calculated as the mean of the samples in a cluster.

        If not set, CBLOF uses KMeans for scalability. See
        https://scikit-learn.org/stable/modules/
            generated/sklearn.cluster.KMeans.html

    alpha : float in (0.5, 1), optional (default=0.9)
        Coefficient for deciding small and large clusters. The ratio
        of the number of samples in large clusters to the number of samples in
        small clusters.

    beta : int or float in (1,), optional (default=5).
        Coefficient for deciding small and large clusters. For a list
        sorted clusters by size `|C1|, |C2|, ..., |Cn|, beta = |Ck|/|Ck-1|`

    use_weights : bool, optional (default=False)
        If set to True, the size of clusters are used as weights in
        outlier score calculation.

    check_estimator : bool, optional (default=False)
        If set to True, check whether the base estimator is consistent with
        sklearn standard.

    random_state : int, RandomState or None, optional (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    Attributes
    ----------
    data_holder : sberpm.DataHolder
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
    >>> from sberpm.ml.anomaly_detection import OutlierCBLOF
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ... 'id_column': [1, 1, 2],
    ... 'activity_column':['st1', 'st2', 'st1'],
    ... 'dt_column':[123456, 123457, 123458]})
    >>> dh = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> outlier_detector = OutlierCBLOF(dh).
    ...     add_groupby_feature('max_time', dh.duration_column, max)
    >>> outlier_detector.apply()
    >>>
    >>> outlier_detector.get_outlier_ids()
    >>> outlier_detector.print_result()
    >>> outlier_detector.show_permutation_importance()
    """

    if TYPE_CHECKING:
        import sberpm

    def __init__(
        self,
        data_holder: sberpm.DataHolder,
        n_clusters: Optional[int] = 8,
        contamination: Optional[float] = 0.1,
        clustering_estimator: Optional[Callable] = None,
        alpha: Optional[float] = 0.9,
        beta: Union[int, float] = 5,
        use_weights: Optional[bool] = False,
        check_estimator: Optional[bool] = False,
        random_state: Optional[int, Callable] = None,
        n_jobs: Optional[int] = 1,
    ):
        super(OutlierCBLOF, self).__init__(data_holder)
        try:
            from pyod.models.cblof import CBLOF

            self._estimator = CBLOF(
                n_clusters=n_clusters,
                contamination=contamination,
                clustering_estimator=clustering_estimator,
                alpha=alpha,
                beta=beta,
                use_weights=use_weights,
                check_estimator=check_estimator,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            self._outlier_label = 1
        except ModuleNotFoundError as e:
            raise PyodModuleNotFoundError() from e
