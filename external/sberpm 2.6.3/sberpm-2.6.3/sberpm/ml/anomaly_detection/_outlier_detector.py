from __future__ import annotations

from typing import Callable, Iterable, Optional, Type, Union

from numpy import array, mean
from numpy import round as round_
from numpy import unique
from pandas import DataFrame, concat

from matplotlib.pyplot import figure, show, title
from seaborn import barplot

from sberpm._holder import DataHolder
from sberpm.ml.utils import PermutationImportance


class OutlierDetector:
    """
    Base class for Inheritance

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names of
            its necessary columns.

    Attributes
    ----------
    _data_holder : DataHolder
        Object that contains the event log and the names of
            its necessary columns.

    _feature_df : DataFrame
        Index: ids of the event traces, columns: features of the event traces.

    _estimator : object
        Algorithm that will be used to detect outliers. It must have either
        both "fit" and "predict" methods or "fit_predict" method.

    _outlier_label : int
        A label that the estimator will use to mark outliers.

    _pred : array-like, shape=[number of event traces]
        Predicted labels for the event traces.

    _permutation_importance_algo : object
        Algorithm used to calculate feature importance that influence an object
        to be an outlier or not.

    """

    def __init__(self, data_holder: Type[DataHolder]):
        self._data_holder = data_holder
        self._feature_df = self._calculate_basic_features()

        self._estimator = None
        self._outlier_label = None
        self._pred = None
        self._permutation_importance_algo = None

    def add_groupby_feature(
        self, feature_name: str, col: str, func: Union[str, Callable]
    ) -> Type[OutlierDetector]:
        """
        Add a feature for detecting outliers.

        Parameters
        ----------
        feature_name : str
            Name of the feature.
        col : str
            Name of the column that the feature will be created from.
        func : str or callable
            Function that will be applied to given column
                after aggregating the data.
        """
        grouped_data = self._data_holder.data.groupby(self._data_holder.id_column, as_index=True)
        self._feature_df = self._feature_df.join((grouped_data.agg({col: func})[col]).rename(feature_name))
        self._feature_df[feature_name] = self._feature_df[feature_name].fillna(0)

        return self

    def add_feature(self, feature_name: str, feature_data: Iterable) -> Type[OutlierDetector]:
        """
        Add a feature for detecting outliers.

        Parameters
        ----------
        feature_name : str
            Name of the feature.
        feature_data : array-like, length = number of IDs in data_holder
            Data that will is considered to be a feature.
        """
        self._feature_df[feature_name] = feature_data
        self._feature_df[feature_name] = self._feature_df[feature_name].fillna(0)
        return self

    def _check_exist_anomaly(self) -> Optional[int]:
        if len(unique(self._pred)) == 1:
            print("Algorithm doesn't find anomaly")
            return 1

    def apply(self) -> None:
        """
        Launches the outlier detection.

        Estimator must already be set.
        """
        if self._estimator is None:
            raise RuntimeError("Set an estimator first")
        if hasattr(self._estimator, "fit_predict"):
            self._pred = array(self._estimator.fit_predict(self._feature_df))
        else:
            self._estimator.fit(self._feature_df)
            self._pred = array(self._estimator.predict(self._feature_df))
        self._check_exist_anomaly()

    def get_outlier_ids(self) -> array:
        """
        Returns the IDs of the outliers.

        Returns
        -------
        ids : array-like
            IDs of the outliers.
        """
        if self._pred is None:
            raise RuntimeError('call "outlier_detector.apply()" first')
        if self._check_exist_anomaly():
            return 0
        outlier_df = self._feature_df[self._pred == self._outlier_label]
        return array(outlier_df.index)

    def add_anomaly_column(
        self, data_holder: Type[DataHolder], name_column: str = "Anomaly_column"
    ) -> Type[DataHolder]:
        """
        Returns DataHolder with added anomaly column.
        Parameters
        ----------
        data_holder: DataHolder
        name_column: new column

        Returns
        -------
            DataHolder with new column
        """
        if data_holder.grouped_data is None:
            id_column = data_holder.id_column
            data_holder.grouped_data = DataFrame({id_column: data_holder.data[id_column].unique()})

        data_holder.grouped_data[name_column] = self._pred == self._outlier_label
        return data_holder

    def print_result(self) -> Union[DataFrame, int]:
        """
        Prints averages of basic features for both outliers and non-outliers.
        """
        if self._pred is None:
            raise RuntimeError('call "outlier_detector.apply()" first')
        if self._check_exist_anomaly():
            return 0
        outlier_result = self._calc_result(self._feature_df[self._pred == self._outlier_label])
        normal_result = self._calc_result(self._feature_df[self._pred != self._outlier_label])
        outlier_result.index = ["outlier"]
        normal_result.index = ["normal"]
        result = concat([outlier_result, normal_result])

        result["percent"] = round_(result["number"] / result["number"].sum() * 100, 2)
        result = result[[result.columns[0], "percent"] + list(result.columns[1:-1])]
        return result.dropna(axis=1)

    def get_permutation_importance(
        self,
        use_calculated_if_exists: bool = True,
        scoring: str = "f1",
        n_iter: int = 5,
        random_state: Optional[int] = None,
    ):
        """
        Returns feature importance using "permutation" method.
        The higher the number for a feature, the more it influences

        Returns
        -------
        feature_importances : array-like of float
            Importances of the features. The order of he features is
                the same as in self.feature_df.
        """
        if self._pred is None:
            raise RuntimeError('call "outlier_detector.apply()" first')
        if not use_calculated_if_exists and self._permutation_importance_algo is not None:
            self._calc_permutation_importance(scoring=scoring, n_iter=n_iter, random_state=random_state)
        return self._permutation_importance_algo.feature_importances_

    def show_permutation_importance(
        self,
        use_calculated_if_exists: bool = True,
        scoring: str = "f1",
        n_iter: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Shows feature importances, calculated using "permutation" method,
            in a bar plot.
        """
        if self._pred is None:
            raise RuntimeError('call "outlier_detector.apply()" first')
        if self._check_exist_anomaly():
            return None
        if self._permutation_importance_algo is None or not use_calculated_if_exists:
            self._calc_permutation_importance(scoring=scoring, n_iter=n_iter, random_state=random_state)

        feat_imp_df = DataFrame(self._permutation_importance_algo.feature_importances_, columns=["Importance"])
        feat_imp_df["Features"] = self._feature_df.columns
        figure(figsize=(15, 5))
        barplot(x="Importance", y="Features", data=feat_imp_df.sort_values(by="Importance", ascending=False))
        title(f"Feature Importance: {self._estimator.__class__.__name__}")
        show()

    def _calc_permutation_importance(self, scoring, n_iter: int, random_state: int) -> None:
        """
        Calculates feature importances.
        """
        self._permutation_importance_algo = PermutationImportance(
            estimator=self._estimator, scoring=scoring, n_iter=n_iter, random_state=random_state
        )
        self._permutation_importance_algo.fit(self._feature_df, self._pred)

    def _calculate_basic_features(self) -> DataFrame:
        """
        Creates basic features for outlier detection.

        Returns
        -------
        feature_df : DataFrame, shape=[number of IDs in data_holder,?]
            Features
        """
        grouped_data = self._data_holder.data.groupby(self._data_holder.id_column)

        features_list = [grouped_data.agg(trace_length=(self._data_holder.activity_column, "count"))]
        if self._data_holder.user_column is not None:
            features_list.append(grouped_data.agg(unique_user_num=(self._data_holder.user_column, "nunique")))
        if self._data_holder.duration_column in self._data_holder.data.columns:
            features_list.append(grouped_data.agg(total_time=(self._data_holder.duration_column, "sum")))

        feature_df = features_list[0]
        for el in features_list[1:]:
            feature_df = feature_df.join(el, on=self._data_holder.id_column)
        return feature_df

    @staticmethod
    def _check_estimator(estimator) -> bool:
        """
        Checks whether a given estimator is valid
        (has either both "fit" and "predict" methods or "fit_predict" method).

        Returns
        -------
        is_valid : bool
            Returns True if given estimator is valid, false otherwise.
        """
        fit = getattr(estimator, "fit", None)
        predict = getattr(estimator, "predict", None)
        fit_predict = getattr(estimator, "fit_predict", None)

        if fit is not None and predict is not None and callable(fit) and callable(predict):
            return True
        if fit_predict is not None and callable(fit_predict):
            return True
        return False

    @staticmethod
    def _calc_result(df: DataFrame) -> DataFrame:
        """
        Calculates averages of basic features.

        Parameters
        ----------
        df : DataFrame
            Features of either outlier or non-outlier objects

        Returns
        ----------
        result : DataFrame
            Averages of basic features.
        """
        number = len(df)
        mean_trace_length = round_(mean(df["trace_length"]))
        mean_unique_user_num = round_(mean(df["unique_user_num"])) if "unique_user_num" in df.columns else None
        mean_total_time = round_(mean(df["total_time"])) if "total_time" in df.columns else None
        return DataFrame(
            {
                "number": [number],
                "trace length": [mean_trace_length],
                "unique users": [mean_unique_user_num],
                "mean time": [mean_total_time],
            }
        )


class PyodModuleNotFoundError(ModuleNotFoundError):
    def __init__(self):
        ModuleNotFoundError.__init__(
            self,
            'This method requires "pyod" package that '
            "is not installed. "
            "Consider using other algorithms instead.",
        )
