from __future__ import annotations

from typing import Union, TYPE_CHECKING, Callable, Optional

from sberpm.ml.anomaly_detection._outlier_detector import OutlierDetector, PyodModuleNotFoundError


class OutlierOCSVM(OutlierDetector):
    """
    Detects outliers using One-Class SVM and gives basic information about them.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names
            of its necessary columns.

    Parameters
    ----------
    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    nu : float, optional
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then 1/n_features will be used instead.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, optional
        Tolerance for stopping criterion.

    shrinking : bool, optional
        Whether to use the shrinking heuristic.

    cache_size : float, optional
        Specify the size of the kernel cache (in MB).

    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

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
    >>> from sberpm.ml.anomaly_detection import OutlierOCSVM
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ... 'id_column': [1, 1, 2],
    ... 'activity_column':['st1', 'st2', 'st1'],
    ... 'dt_column':[123456, 123457, 123458]})
    >>> dh = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> outlier_detector = OutlierOCSVM(dh).
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
        kernel: Union[str, Callable] = "rbf",
        degree: int = 3,
        gamma: Optional[float, str] = "auto",
        coef0: Optional[float] = 0.0,
        tol: Optional[float] = 1e-3,
        nu: Optional[float] = 0.5,
        shrinking: bool = True,
        cache_size: Optional[float] = 200,
        verbose: bool = False,
        max_iter: Optional[int] = -1,
        contamination: Optional[float] = 0.1,
    ):

        super(OutlierOCSVM, self).__init__(data_holder)
        try:
            from pyod.models.ocsvm import OCSVM

            self._estimator = OCSVM(
                kernel=kernel,
                degree=degree,
                gamma=gamma,
                coef0=coef0,
                tol=tol,
                nu=nu,
                shrinking=shrinking,
                cache_size=cache_size,
                verbose=verbose,
                max_iter=max_iter,
                contamination=contamination,
            )
            self._outlier_label = 1
        except ModuleNotFoundError as err:
            raise PyodModuleNotFoundError() from err
