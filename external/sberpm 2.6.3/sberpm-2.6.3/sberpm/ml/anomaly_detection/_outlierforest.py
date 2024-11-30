from typing import Type, Union, Optional, Callable

from sklearn.ensemble import IsolationForest

from sberpm.ml.anomaly_detection._outlier_detector import OutlierDetector

from sberpm._holder import DataHolder


class OutlierForest(OutlierDetector):
    """
    Detects outliers using Isolation Forest and gives
        basic information about them.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names
            of its necessary columns.

    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range [0, 0.5].

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    random_state : int or RandomState, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

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
    >>> from sberpm.ml.anomaly_detection import OutlierForest
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ... 'id_column': [1, 1, 2],
    ... 'activity_column':['st1', 'st2', 'st1'],
    ... 'dt_column':[123456, 123457, 123458]})
    >>> dh = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> outlier_detector = OutlierForest(dh).
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
        n_estimators: int = 100,
        max_samples: Union[str, int, float] = "auto",
        contamination: Union[str, float] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Union[int, Callable] = None,
        verbose: int = 0,
        warm_start: bool = False,
    ):
        super(OutlierForest, self).__init__(data_holder)
        self._estimator = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )
        self._outlier_label = -1
