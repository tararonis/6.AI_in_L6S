from __future__ import annotations

from importlib import import_module
from typing import Type, List, Callable
from dataenforce import Dataset

from numpy import array
from pyod.models.combination import majority_vote
from sklearn.preprocessing import StandardScaler

from sberpm.ml.anomaly_detection._outlier_detector import OutlierDetector
from sberpm._holder import DataHolder


class _OutlierAlgorithmsWrapper:
    def __init__(self, algorithms: List[Callable]):
        self._algorithms = algorithms
        self._is_fitted = False
        self._result_labels = None

    # TODO Dataset type
    def fit(self, feature_df: Dataset[str]) -> "_OutlierAlgorithmsWrapper":
        assert not self._is_fitted, "Already fitted!"
        scaled_feature_df = StandardScaler().fit_transform(feature_df)
        full_labels = []
        for algorithm in self._algorithms:
            fitted = algorithm().fit(scaled_feature_df)
            full_labels.append(fitted.labels_)

        full_labels = array(full_labels).T
        self._result_labels = majority_vote(full_labels)
        self._is_fitted = True
        return self

    def predict(self) -> array:
        assert self._is_fitted, "Please use fit before predict!"
        return self._result_labels

    # TODO Dataset type
    def fit_predict(self, feature_df: Dataset[str]) -> array:
        if self._is_fitted:
            return self._result_labels
        scaled_feature_df = StandardScaler().fit_transform(feature_df)
        full_labels = []
        for algorithm in self._algorithms:
            if str(algorithm.mro()[0]) == "<class 'pyod.models.iforest.IForest'>":
                fitted = algorithm(random_state=42).fit(scaled_feature_df)
            else:
                fitted = algorithm().fit(scaled_feature_df)
            full_labels.append(fitted.labels_)

        full_labels = array(full_labels).T
        self._result_labels = majority_vote(full_labels)
        self._is_fitted = True
        return self._result_labels


class OutlierEnsemble(OutlierDetector):
    """
    Detects outliers using multiple algorithms
    and gives basic information about them.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names
            of its necessary columns.
    algorithms: List[str]
        List of algorithms for majority voting for outliers.
        Supports these algorithms from pyod.models: [
            "KNN",
            "IForest",
            "HBOS",
            "ABOD"
        ]

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
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.ml.anomaly_detection import OutlierEnsemble
    >>> df = pd.DataFrame({
    ... 'id_column': [1, 1, 2],
    ... 'activity_column':['st1', 'st2', 'st1'],
    ... 'dt_column':[123456, 123457, 123458]})
    >>> dh = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>> outlier_detector = OutlierEnsemble(data_holder=dh,
    ...     algorithms=["HBOS", "ABOD", "KNN", "IForest"])
    >>> outlier_detector.apply()
    >>> outlier_detector.get_outlier_ids()
    >>> outlier_detector.print_result()
    """

    def __init__(self, data_holder: Type[DataHolder], algorithms: List[str]):

        assert len(set(algorithms) - {"KNN", "IForest", "HBOS", "ABOD"}) == 0, (
            f"Outlier detection algorithms should be "
            f"from ['KNN', 'IForest', 'HBOS', 'ABOD'], but get {algorithms}"
        )
        super(OutlierEnsemble, self).__init__(data_holder)
        self._algorithms = []
        if "KNN" in algorithms:
            self._algorithms.append(getattr(import_module("pyod.models.knn"), "KNN"))
        if "IForest" in algorithms:
            self._algorithms.append(getattr(import_module("pyod.models.iforest"), "IForest"))
        if "HBOS" in algorithms:
            self._algorithms.append(getattr(import_module("pyod.models.hbos"), "HBOS"))
        if "ABOD" in algorithms:
            self._algorithms.append(getattr(import_module("pyod.models.abod"), "ABOD"))
        self._estimator = _OutlierAlgorithmsWrapper(self._algorithms)
        self._outlier_label = 1
