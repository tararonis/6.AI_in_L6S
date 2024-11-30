from typing import Type, Callable

from sberpm._holder import DataHolder
from sberpm.ml.anomaly_detection._outlier_detector import OutlierDetector


class OutlierCustom(OutlierDetector):
    """
    Detects outliers with Custom Outlier detector and gives
        basic information about them.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names
            of its necessary columns.

    estimator: object
        Algorithm that will be used to detect outliers. It must have either
        both "fit" and "predict" methods or "fit_predict" method.

    outlier_label : int
        A label that the estimator will use to mark outliers.

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
    >>> from sberpm.ml.anomaly_detection import OutlierCustom
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ... 'id_column': [1, 1, 2],
    ... 'activity_column':['st1', 'st2', 'st1'],
    ... 'dt_column':[123456, 123457, 123458]})
    >>> dh = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>> estimator =
    >>> outlier_detector = OutlierCustom(dh,estimator).
    ...     add_groupby_feature('max_time', dh.duration_column, max)
    >>> outlier_detector.apply()
    >>>
    >>> outlier_detector.get_outlier_ids()
    >>> outlier_detector.print_result()
    >>> outlier_detector.show_permutation_importance()
    """

    def __init__(self, data_holder: Type[DataHolder], estimator: Callable, outlier_label: int):
        super(OutlierCustom, self).__init__(data_holder)

        if self._check_estimator(estimator):
            self._estimator = estimator
            self._outlier_label = outlier_label
        else:
            raise AttributeError(
                "An estimator object must have either "
                'both "fit" and "predict" methods '
                'or "fit_predict" method.'
            )
