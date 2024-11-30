from itertools import product

from colorama import Fore, Style

from numpy import argmax, argwhere, array, count_nonzero, delete, exp, linspace, log1p, quantile
from pandas import DataFrame, concat, get_dummies
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from imblearn import over_sampling, under_sampling
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree


from matplotlib.pyplot import savefig as savefigure
from matplotlib.pyplot import figure, subplots, suptitle, title, xlabel, ylabel, yticks
from seaborn import barplot, heatmap, histplot


class DecisionMining:
    """
    Class that performs decision point analysis, also referred to as decision mining.

    Parameters
    ----------
    data_holder: sberpm.DataHolder
        Object that contains the event log and names of its necessary columns
        (id, activities, timestamps, etc.).

    Examples
    --------
    >>> from sberpm.decision_mining import DecisionMining
    >>> dm = DecisionMining(data_holder)
    >>> dm.apply(decision_points='all', categorical_attrs= [data_holder.user_column],
    >>>     noncategorical_attrs=[data_holder.duration_column])
    >>> dm.plot_decision_tree(decision_points='all')
    >>> dm.print_decision_rule(decision_points='all')
    >>> dm.plot_feature_distribution(decision_points='all', clf_results=True)
    """

    def __init__(self, data_holder):
        data_holder.check_or_calc_duration()
        data_holder.get_grouped_data(data_holder.activity_column, data_holder.duration_column)
        self._threshold = 0.001
        self._points = self._decision_points(data_holder)
        self._holder = data_holder

        self._numeric_attributes = []
        self._categorical_attributes = []
        self._boolean_attributes = []

        self._trees = {}
        self._confusion_matrix = {}
        self._feature_importances = {}
        self._metrics = {}  # key: decision point, value: dict of metric: value

        self._data_origin = None
        self._attributes_origin = None
        self._data = None
        self._attributes = None
        self._data_pred = None
        self._pred_idx = []

        self._str_tuple_attrs_dict = None

    def _decision_points(self, data_holder):
        """
        Identifies decision points, that is parts of the model where the process is split into alternative branches.
        Rare cases are not taken into account.

        Parameters
        ----------
        data_holder: sberpm.DataHolder

        Returns
        -------
        df: pandas.DataFrame
            Decision points and decisions that have been made, columns: [decision point (str), class (str)]
        """
        df = DataFrame(
            zip(
                data_holder.data[data_holder.activity_column],
                data_holder.data[data_holder.activity_column].shift(-1),
            ),
            columns=["Decision Point", "Class"],  # TODO dups
        )
        id_change_mask = (
            data_holder.data[data_holder.id_column].shift(-1) != data_holder.data[data_holder.id_column]
        )
        df.drop(
            df.index[id_change_mask], axis=0, inplace=True
        )  # no reset of indexes, leave as in original data_holder

        df_ = df.groupby(["Decision Point", "Class"]).size().unstack(fill_value=0)
        not_points = df_.index[count_nonzero(df_, axis=1) == 1]
        df_.drop(not_points, inplace=True)

        to_consider = []
        sum_by_point = df_.sum(axis=1)
        for i in product(df_.index, df_.columns):
            threshold = sum_by_point[i[0]] * self._threshold
            if df_.loc[i] > threshold and df_.loc[i] > 1:
                to_consider.append(i)
            else:
                df_.loc[i] = 0

        not_points = df_.index[count_nonzero(df_, axis=1) == 1]
        df_.drop(not_points, inplace=True)

        df["Pair"] = tuple(zip(df.iloc[:, 0], df.iloc[:, 1]))
        df = df[df["Pair"].isin(to_consider)].drop(["Pair"], axis=1)
        df = df[df["Decision Point"].isin(df_.index)]
        return df

    # TODO refactor
    def apply(
        self,
        categorical_attrs=None,
        noncategorical_attrs=None,
        decision_points="all",
        sampling=None,
        tree_params="default",
        grid_search=False,
        param_grid="default",
        random_state=42,
        n_jobs=None,
    ):
        """
        Performs decision mining for a given set of decision points based on the data attributes contained in the log
        by constructing a decision tree classifier.

        Parameters
        ----------
        categorical_attrs: list of str
            Names of the columns of categorical values that will be used as  for rules creation.
            The columns will be converted to str type.
            Note: at least one of "categorical_attrs" and "noncategorical_attrs" must not be None.

        noncategorical_attrs: list of str
            Names of the columns of non-categorical values that will be used as  for rules creation.
            Columns might be of numeric, boolean and Timedelta types.
            Note: at least one of "categorical_attrs" and "noncategorical_attrs" must not be None.

        decision_points: list of str, default='all'
            Points which decision mining will be performed for.

        sampling: {'RandomOverSampler', 'RandomUnderSampler'} or object, default=None
            If not None, an algorithm for balancing classes to avoid the problem of imbalanced data set.
            If str, the name of an algorithm from imbalanced-learn package (imblearn).
            If object, an algorithm that must have a "fit_resample" method (like the algorithms from imblearn package).

        tree_params: dict of {str: str or int}, default='default'
            Parameters of the sklearn decision tree classifier.

        grid_search: bool or dict, default=False
            Whether to tune the hyper-parameters of the decision tree classifier via grid search.
            If False, grid search is not used.
            If True, grid search is used with the default parameters:
                scoring='f1_weighted', cv=4, verbose=5
            If dict, it is used as parameters for the algorithm directly.
            Note: dict must not contain "estimator" and "param_grid" arguments.

        param_grid: dict, default='default'
            This argument is used only if grid search is used.
            If 'default', the following param grid is used:
                'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2'] + [None],
                'max_depth': [int(x) for x in linspace(10, 50, 5)] + [None],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 5, 8, 12].
            If dict, used as a param_grid directly.

        random_state: int, default=42
            Used in sampling and tree classifier algorithms.
            If tree_params is a dict and has 'random_state' param, the one from the dict is used.

        n_jobs: int, default=None
            Used in sampling where possible and grid_search algorithms.
            If grid_search is a dict and has 'n_jobs' param, the one from the dict is used.
        """
        if categorical_attrs is None and noncategorical_attrs is None:
            raise RuntimeError('Both "categorical_attrs" and "noncategorical_attrs" cannot be None.')

        all_attrs = []
        for attrs_i in (categorical_attrs, noncategorical_attrs):
            if attrs_i is not None:
                all_attrs += attrs_i

        data = concat([self._points, self._holder.data[all_attrs].loc[self._points.index]], axis=1)

        # Separate columns by types and fillna
        self._categorical_attributes = []
        self._numeric_attributes = []
        self._boolean_attributes = []

        # Categorical
        if categorical_attrs is not None:
            for col in categorical_attrs:
                data[col] = data[col].astype(str)  # null values will become 'nan'
            self._categorical_attributes = categorical_attrs

        # Non-categorical
        if noncategorical_attrs is not None:
            for col in noncategorical_attrs:
                if is_bool_dtype(data[col]):  # (can't have None values?) (bool is also numeric dtype)
                    self._boolean_attributes.append(col)
                elif is_numeric_dtype(data[col]):
                    self._numeric_attributes.append(col)
                    data.fillna({col: data[col].min() - 1}, inplace=True)
                elif is_datetime64_any_dtype(data[col]):
                    time_series = data[col]
                    nonna_mask = ~time_series.isna()
                    time_series = time_series[nonna_mask]
                    new_cols = {
                        f"{col}_Day-of-week": lambda x: x.dayofweek + 1,
                        f"{col}_Day": lambda x: x.day,
                        f"{col}_Month": lambda x: x.month,
                        f"{col}_Year": lambda x: x.year,
                    }
                    for ncol, func in new_cols.items():
                        data[ncol] = -1
                        data.loc[nonna_mask, ncol] = time_series.apply(func)
                    self._numeric_attributes += list(new_cols.keys())
                    data.drop(col, axis=1, inplace=True)
                else:
                    raise TypeError(
                        "Columns of numeric, bool and datetime dtypes only are supported in "
                        f'"noncategorical_attrs", but got column "{col}" of dtype "{data[col].dtype}"'
                    )

        self._data_origin = data
        self._attributes_origin = list(data.columns[2:])
        if len(self._categorical_attributes) != 0:
            self._str_tuple_attrs_dict = {
                f"{col}_{val}": (col, val) for col in self._categorical_attributes for val in data[col].unique()
            }
            data = get_dummies(data, columns=self._categorical_attributes, prefix_sep="_", drop_first=False)
        self._attributes = list(data.columns[2:])
        self._data = data

        self._data_pred = self._data_origin.copy()
        self._pred_idx = []

        if decision_points == "all":
            decision_points = self._points["Decision Point"].unique()
        for point in decision_points:
            X = data[data["Decision Point"] == point].iloc[:, 2:]
            y = data[data["Decision Point"] == point]["Class"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
            if sampling is not None:
                if isinstance(sampling, str):
                    if sampling == "RandomOverSampler":
                        algo = over_sampling.RandomOverSampler(random_state=random_state)
                    elif sampling == "RandomUnderSampler":
                        algo = under_sampling.RandomUnderSampler(random_state=random_state)
                    else:
                        raise RuntimeError(
                            "Sampling algorithm of str type must be one of 'RandomOverSampler', "
                            "'RandomUnderSampler', but got '{sampling}'"
                        )
                else:
                    fit_resample = getattr(sampling, "fit_resample", None)
                    if fit_resample is not None and callable(fit_resample):
                        algo = sampling
                    else:
                        raise RuntimeError("Sampling algorithm object must have 'fit_resample' method.")

                if "n_jobs" in set(algo.get_params().keys()):
                    algo.set_params(n_jobs=n_jobs)

                X_train, y_train = algo.fit_resample(X_train, y_train)

            # Tree algorithm
            if tree_params == "default":
                clf = DecisionTreeClassifier(random_state=random_state)
            elif isinstance(tree_params, dict):
                if "random_state" not in tree_params:
                    tree_params["random_state"] = random_state
                clf = DecisionTreeClassifier(**tree_params)
            else:
                raise TypeError(
                    f"tree_params must equal 'default', or be of type dict, "
                    f"but got {type(tree_params)} - {tree_params}"
                )

            # Grid search algorithm
            if not (isinstance(grid_search, bool) and not grid_search):
                # Algo params
                if isinstance(grid_search, bool) and grid_search:
                    params = dict(scoring="f1_weighted", cv=4, verbose=5, n_jobs=n_jobs)
                elif isinstance(grid_search, dict):
                    params = grid_search
                    if "n_jobs" not in params:
                        params["n_jobs"] = n_jobs
                else:
                    raise TypeError(
                        f"grid_search_params must equal 'default', be None, or be of type dict, "
                        f"but got {type(grid_search)} - {grid_search}"
                    )
                # Clf param grid
                if param_grid == "default":
                    grid = {
                        "criterion": ["gini", "entropy"],
                        "max_features": ["sqrt", "log2"] + [None],
                        "max_depth": [int(x) for x in linspace(10, 50, 5)] + [None],
                        "min_samples_split": [2, 5, 10, 15, 20],
                        "min_samples_leaf": [1, 2, 5, 8, 12],
                    }
                elif isinstance(param_grid, dict):
                    grid = param_grid
                else:
                    raise TypeError(
                        f"param_grid must equal 'default', or be of type dict, "
                        f"but got {type(param_grid)} - {param_grid}"
                    )
                gsearch = GridSearchCV(estimator=clf, param_grid=grid, **params)
                gsearch.fit(X_train, y_train)
                clf = gsearch.best_estimator_
            else:
                clf.fit(X_train, y_train)
            self._trees[point] = clf
            y_pred = clf.predict(X_test)
            self._confusion_matrix[point] = confusion_matrix(y_test, y_pred)
            self._feature_importances[point] = list(zip(self._attributes, clf.feature_importances_))
            self._data_pred.loc[X_test.index, "Class"] = y_pred
            self._pred_idx.extend(X_test.index.values)

            if clf.n_classes_ == 2:
                self._metrics[point] = dict(
                    accuracy=accuracy_score(y_test, y_pred),
                    precision_binary=precision_score(
                        y_test, y_pred, pos_label=sorted(clf.classes_)[0], average="binary"
                    ),
                    recall_binary=recall_score(
                        y_test, y_pred, pos_label=sorted(clf.classes_)[0], average="binary"
                    ),
                    f1_score_binary=f1_score(y_test, y_pred, pos_label=sorted(clf.classes_)[0], average="binary"),
                    precision_micro=precision_score(y_test, y_pred, average="micro"),
                    recall_micro=recall_score(y_test, y_pred, average="micro"),
                    f1_score_micro=f1_score(y_test, y_pred, average="micro"),
                    precision_macro=precision_score(y_test, y_pred, average="macro"),
                    recall_macro=recall_score(y_test, y_pred, average="macro"),
                    f1_score_macro=f1_score(y_test, y_pred, average="macro"),
                    precision_weighted=precision_score(y_test, y_pred, average="weighted"),
                    recall_weighted=recall_score(y_test, y_pred, average="weighted"),
                    f1_score_weighted=f1_score(y_test, y_pred, average="weighted"),
                )
            else:
                self._metrics[point] = dict(
                    accuracy=accuracy_score(y_test, y_pred),
                    precision_micro=precision_score(y_test, y_pred, average="micro"),
                    recall_micro=recall_score(y_test, y_pred, average="micro"),
                    f1_score_micro=f1_score(y_test, y_pred, average="micro"),
                    precision_macro=precision_score(y_test, y_pred, average="macro"),
                    recall_macro=recall_score(y_test, y_pred, average="macro"),
                    f1_score_macro=f1_score(y_test, y_pred, average="macro"),
                    precision_weighted=precision_score(y_test, y_pred, average="weighted"),
                    recall_weighted=recall_score(y_test, y_pred, average="weighted"),
                    f1_score_weighted=f1_score(y_test, y_pred, average="weighted"),
                )

    def get_clf_metrics(self):
        """
        Calculates classification metrics to measure model performance.

        Returns
        -------
        df: pandas.DataFrame
            Classification metrics for each decision point.
        """
        return concat([DataFrame({d_point: metrics}) for d_point, metrics in self._metrics.items()], axis=1)

    # TODO refactor
    def print_decision_rule(self, decision_points="all", paths="all"):
        """
        Returns decision rules (logical expressions) for given decision points and paths.

        Parameters
        ----------
        decision_points: list of str, default='all'
            Points which decision rules will be formulated for.

        paths: list of str, default='all'
            Paths which decision rules will be formulated for.

        Returns
        -------
        report: str
            Text summary of all the rules in the decision tree.
        """
        message = ""
        if decision_points == "all":
            decision_points = self._points["Decision Point"].unique()
        cat_values_dict = {}
        if len(self._categorical_attributes) != 0:
            for feature in self._categorical_attributes:
                cat_values_dict[feature] = self._data_origin[feature].unique()
        for point in decision_points:
            message += f"From decision point: {point}\n"
            rules_dict = self._tree_to_rules(self._trees[point], self._attributes)
            paths2 = rules_dict.keys() if paths == "all" else paths
            for i, key in enumerate(paths2, start=1):
                message += f"to {Fore.BLUE} {key} {Style.RESET_ALL} if:\n"
                message += f"{Style.RESET_ALL}"
                for rules in rules_dict[key]:
                    cat_values_dict_temp = cat_values_dict.copy()
                    (
                        cat_rules,
                        noncat_rules,
                    ) = self._rules_for_categorical(rules, cat_values_dict_temp)
                    str_rules = [
                        f'{r[0]}_({(Fore.MAGENTA + " OR " + Style.RESET_ALL).join(r[1])})'
                        if len(r[1]) > 1
                        else f"{r[0]}_{r[1]}"
                        for r in cat_rules
                    ]
                    str_rules += [f"{r[0]} {r[1]} {r[2]} " for r in noncat_rules]
                    message += f"{Fore.RED} AND {Style.RESET_ALL}".join(str_rules)
                    if i < len(rules_dict[key]):
                        message += f"{Fore.GREEN} OR {Style.RESET_ALL}\n"
        print(message)

    def print_decision_points(self):
        """
        Identifies decision points in a process model.

        Returns
        -------
        result: str
            Decision points and different decisions that can be made.
        """
        df = self._points.groupby(["Decision Point"]).agg({"Class": set}).reset_index()
        for point in df["Decision Point"].unique():
            print(point, "->")
            print(*df[df["Decision Point"] == point]["Class"].values[0], sep=" / ")
            print()

    def _rules_for_categorical(self, rules, dict_cat):
        """
        Separate categorical rules from the other ones (numeric and bool).

        Parameters
        ----------
        rules: list of (str, str, {number, or bool})
            Decision rules.
            Every separate rule is a tuple of three parameters:
            feature name, sign {'=', '<=', '>'}, and a value.

        dict_cat: dict of {str: numpy.array}
            Categories that features they can take.

        Returns
        -------
        cat_rules: list of (str, list of str)
            Decision rules adjusted for categories.
            Every separate rule is a tuple of two parameters:
            feature name, and a list of its values.

        noncat_rules: list of (str, str, {number, or bool})
            Non-categorical decision rules.
            Every separate rule is a tuple of three parameters:
            feature name, sign {'=', '<=', '>'}, and a value.
        """
        noncat_rules = []

        for (name, sign, value) in rules:
            if name in self._categorical_attributes:
                feature, category = self._str_tuple_attrs_dict[name]
                if int(value) == 1:
                    dict_cat[feature] = array([category], dtype=object)
                else:
                    dict_cat[feature] = delete(dict_cat[feature], argwhere(dict_cat[feature] == category))
            else:
                noncat_rules.append((name, sign, value))
        cat_rules = list(dict_cat.items())
        return cat_rules, noncat_rules

    # TODO refactor
    def _tree_to_rules(self, decision_tree, feature_names):
        """
        Extracts the rules from a decision tree.

        Parameters
        ----------
        decision_tree: sklearn.tree
            Decision tree.

        feature_names: list of str
            Feature names.

        Returns
        -------
        paths: dict of {str: list of (str, str, {number, or bool})}
            Decision rules.
            Every separate rule is a tuple of three parameters:
            feature name, sign {'=', '<=', '>'}, and a value.
        """
        tree_ = decision_tree.tree_
        class_names = decision_tree.classes_
        feature_names_ = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
        paths = {}
        rules = []

        def tree_recurse(node, depth):
            if tree_.n_outputs == 1:
                value = tree_.value[node][0]
            else:
                value = tree_.value[node].T[0]
            class_name = argmax(value)
            if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
                class_name = class_names[class_name]

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names_[node]
                threshold = tree_.threshold[node]
                if name in self._boolean_attributes:
                    rules.append((name, "=", False))
                elif name in self._numeric_attributes:
                    rules.append((name, "<=", threshold))
                else:
                    rules.append((name, "=", 0))
                tree_recurse(tree_.children_left[node], depth + 1)
                del rules[depth:]
                if name in self._boolean_attributes:
                    rules.append((name, "=", True))
                elif name in self._numeric_attributes:
                    rules.append((name, ">", threshold))
                else:
                    rules.append((name, "=", 1))
                tree_recurse(tree_.children_right[node], depth + 1)
            else:
                if class_name not in paths:
                    paths[class_name] = []
                paths[class_name].append(list(rules))

        tree_recurse(0, 0)
        return paths

    def plot_confusion_matrix(self, decision_points="all", savefig=False):
        """
        Plots the confusion matrices for given decision points.

        Parameters
        ----------
        decision_points: list of str, default='all'
            Points which confusion matrices will be plotted for.

        savefig: bool, default=False
            Whether to save the figure.

        Returns
        -------
        plot: seaborn.heatmap
            Visualized confusion matrices.
        """
        if decision_points == "all":
            decision_points = self._points["Decision Point"].unique()
        for point in decision_points:
            figure()
            heatmap(
                DataFrame(
                    self._confusion_matrix[point],
                    columns=self._trees[point].classes_,
                    index=self._trees[point].classes_,
                ),
                annot=True,
                fmt=".0f",
                cmap="coolwarm",
                linewidths=0.5,
            )

            title(f"Decision Point: {point}", weight=700, size=14)
            ylabel("True")
            xlabel("Predicted")
            yticks(rotation=0)

            if savefig:
                savefigure(f"confusion_matrix_{point}.pdf", bbox_inches="tight")

    def plot_feature_importance(self, decision_points="all", savefig=False):
        """
        Plots the feature importances for given decision points.

        Parameters
        ----------
        decision_points: list of str, default='all'
            Points which feature importance will be plotted for.

        savefig: bool, default=False
            Whether to save the figure.

        Returns
        -------
        plot: seaborn.barplot
            Visualized feature importances.
        """
        if decision_points == "all":
            decision_points = self._points["Decision Point"].unique()
        for point in decision_points:
            values = self._feature_importances[point]
            fi = DataFrame(values, columns=["Feature", "Importance"])
            figure(figsize=(10, 4))
            barplot(
                x="Importance", y="Feature", data=fi.sort_values(by="Importance", ascending=False), palette="husl"
            )
            title(f"Decision Point: {point}", weight=700, size=14)

            if savefig:
                savefigure(f"feature_importance_{point}.pdf", bbox_inches="tight")

    def plot_decision_tree(self, decision_points="all", max_depth=None, scale=3, savefig=False):
        """
        Plots the decision trees for given decision points.

        Parameters
        ----------
        decision_points: list of str, default='all'
            Points which decision trees will be plotted for.

        max_depth: int, default=None
            The maximum depth of the representation. If None, the tree is fully generated.

        scale: int, default=3
            Scale of the representation.

        savefig: bool, default=False
            Whether to save the figure.

        Returns
        -------
        plot: sklearn.tree.plot_tree
            Visualized decision trees.
        """
        if decision_points == "all":
            decision_points = self._points["Decision Point"].unique()
        for point in decision_points:
            fig = figure(
                figsize=(self._trees[point].get_depth() * (scale + 1), self._trees[point].get_depth() * scale)
            )
            _ = plot_tree(
                self._trees[point],
                max_depth=max_depth,
                feature_names=self._attributes,
                class_names=sorted(self._trees[point].classes_),
                filled=True,
            )

            title(f"Decision Point: {point}", weight=700, size=14)
            if savefig:
                fig.savefigure(f"decision_tree_{point}.pdf", bbox_inches="tight")

    # TODO refactor
    def plot_feature_distribution(
        self, decision_points="all", drop_outliers=False, clf_results=False, savefig=False
    ):
        """
        Plots the feature distributions for given decision points.

        Parameters
        ----------
        decision_points: list of str, default='all'
            Points which feature distributions will be plotted for.

        drop_outliers: bool, default=False
            Whether to drop outliers for numerical features.

        clf_results: bool, default=False
            Whether to show results of classification. If False, distributions are taken from the log.

        savefig: bool, default=False
            Whether to save the figure.

        Returns
        -------
        plot: seaborn.histplot
            Visualized feature distributions.
        """
        if decision_points == "all":
            decision_points = self._points["Decision Point"].unique()
        for point in decision_points:
            _, axes = subplots(
                nrows=1, ncols=len(self._attributes_origin), figsize=(len(self._attributes_origin) * 8, 4)
            )
            i = 0
            if clf_results:
                data = self._data_pred.loc[self._pred_idx][self._data_pred["Decision Point"] == point]
            else:
                data = self._data_origin[self._data_origin["Decision Point"] == point]
            if len(self._categorical_attributes) != 0:
                for attribute in self._categorical_attributes:
                    data.groupby([attribute, "Class"]).size().unstack().plot(
                        kind="bar", ax=axes[i], alpha=0.5, ylabel="Count", xlabel="", title=attribute, legend=True
                    )
                    i += 1
            if len(self._boolean_attributes) != 0:
                for attribute in self._boolean_attributes:
                    data.groupby([attribute, "Class"]).size().unstack().plot(
                        kind="bar", ax=axes[i], alpha=0.5, ylabel="Count", xlabel="", title=attribute, legend=True
                    )
                    i += 1
            if len(self._numeric_attributes) != 0:
                for attribute in self._numeric_attributes:
                    if attribute.endswith(("Day-of-week", "Day", "Month", "Year")):
                        data.groupby([attribute, "Class"]).size().unstack().plot(
                            kind="bar",
                            ax=axes[i],
                            alpha=0.5,
                            ylabel="Count",
                            xlabel="",
                            title=attribute,
                            legend=True,
                        )
                    else:
                        data_ = data.copy()
                        data_[attribute] = log1p(data_[attribute])
                        if drop_outliers:
                            q1 = quantile(data_[attribute], 0.25)
                            q3 = quantile(data_[attribute], 0.75)
                            iqr = q3 - q1
                            q1_mask = data_[attribute] > q1 - 1.5 * iqr
                            q3_mask = data_[attribute] < q3 + 1.5 * iqr
                            data_[attribute] = data_[attribute][q1_mask & q3_mask]
                        z = histplot(
                            data_,
                            bins=30,
                            x=attribute,
                            hue="Class",
                            kde=True,
                            stat="density",
                            common_norm=False,
                            hue_order=sorted(data_["Class"].unique()),
                            alpha=0.4,
                            edgecolor="white",
                            linewidth=0.3,
                            legend=True,
                            ax=axes[i],
                        )
                        z.set_title(attribute)
                        z.set_xlabel("Logarithmic scale")
                        # add second x-axis
                        secax = z.twiny()
                        new_tick_locations = z.get_xticks()
                        secax.xaxis.set_ticks_position("bottom")
                        secax.xaxis.set_label_position("bottom")
                        secax.spines["bottom"].set_position(("outward", 36))
                        secax.set_xticks(new_tick_locations)
                        secax.set_xticklabels(["%d " % z for z in exp(new_tick_locations)])
                        secax.set_xlabel("Initial scale")
                        secax.set_xlim(z.get_xlim())
                        # add second y-axis to second x-axis
                        secax_to_secax = secax.twinx()
                        secax_to_secax.yaxis.tick_left()
                        secax_to_secax.set_yticks(z.get_yticks())
                        secax_to_secax.set_ylim(z.get_ylim())
                        secax_to_secax.yaxis.set_label_position("left")
                        secax_to_secax.set_ylabel("Density")
                    i += 1

            suptitle("Decision Point: " + point, weight=700, size=14)
            if savefig:
                savefigure(f"feature_distribution_{point}.pdf", bbox_inches="tight")
