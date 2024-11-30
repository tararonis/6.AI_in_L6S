from __future__ import annotations

from functools import partialmethod
from typing import TYPE_CHECKING

from dataclassy import dataclass

from numpy import (
    array,
    cumsum,
    exp,
    float64,
    log,
    mean,
    nan_to_num,
    ravel,
    sum as np_sum,
    unique,
    zeros_like,
)
from numpy.random import choice, normal, seed as set_random_state
from pandas import DataFrame, Series, crosstab, to_timedelta

from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

from tqdm import tqdm

from sberpm._holder import DataHolder
from sberpm.metrics import ActivityMetric, TraceMetric, TransitionMetric

if TYPE_CHECKING:
    from numpy import int64
    from numpy.typing import NDArray
    from pandas import Timestamp


set_random_state(42)  # random state for sklearn
tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)  # disable tqdm progress


@dataclass(slots=True)
class Simulation:
    # TODO docstring
    data_holder: DataHolder

    _activity_mutation_data: DataFrame = None
    _action_duration: dict[str, NDArray[float64]] = None
    _cross_tab: DataFrame = None

    _result: DataFrame = None
    _mean_duration: float64 = None
    _start_timestamp_traces: list[
        dict[str, int | float | str | list[str] | Timestamp]
    ] = []

    __inter_start: NDArray = array([])  # TODO subtype, replace array
    __node_dict: dict = {}  # TODO type, defaultdict
    __time_like_nodes: dict = {}  # TODO type, defaultdict

    def __post_init__(self):
        self.data_holder.check_or_calc_duration()

        self._activity_mutation_data = self._assign_end_activity_by_process()
        self._action_duration = self._sample_approx_activities_duration()
        self._cross_tab = self._get_cross_tab()

        self._recalculate_cross_tab_probabilities()  # ? TODO rename

    # TODO refactor
    def generate(self, iterations=100, start_time=None):
        """
        Generate and save process traces, start_time and duration for every action,

        Parameters
        ----------
        iterations : int
            number of traces
        """
        self.__node_dict = self._cross_tab_to_dict()
        self.__inter_start = self._calc_inter_start_rate(iterations - 1)

        initial_start_time = (
            start_time
            if start_time is not None
            else min(self.data_holder.data[self.data_holder.start_timestamp_column])
        )

        # ? TODO tuple
        self._start_timestamp_traces = [
            dict(
                id_column=0,
                trace=self._rec_choice_trace("start", []),
                start=initial_start_time,
                duration=None,
            ),
            *[
                dict(
                    id_column=iteration,
                    trace=self._rec_choice_trace("start", []),
                    start=initial_start_time
                    + to_timedelta(
                        self.__inter_start[iteration - 1],
                        unit="s",
                        errors="coerce",
                    ),
                    duration=None,  # ? TODO move None duration closer to use
                )
                for iteration in tqdm(range(1, iterations))
            ],
        ]

        self._result = DataFrame()  # reset _result every generate run

    def get_result(self):
        """
        Convert result to dataframe (in early not converted) and return result

        Returns
        -------
            results : DataFrame, column as in init data_holder
        """
        if self._result.empty:
            self._convert_result_to_df()

        return self._result

    def swap_nodes(self, node_first, node_second, save_probabilities=True):
        """
        Swap nodes in process graph, swap nodes names, action duration and edge probabilities (optional)

        Parameters
        ----------
            node_first : str
            node_second : str
            save_probabilities : bool
                if False swap nodes and probabilities of pre-edges
        """
        if node_first in ["start", "end"] or node_second in ["start", "end"]:
            return

        if (
            node_first in self._cross_tab.index
            and node_second in self._cross_tab.columns
            and not save_probabilities
        ):
            node_first_probs = self._to_prob(self._cross_tab[node_first].values)
            node_second_probs = self._to_prob(self._cross_tab[node_second].values)
            mean_first = mean(node_first_probs)
            mean_second = mean(node_second_probs)

            for prob_idx, node in enumerate(self._cross_tab.index):
                self._add_edge(
                    node, node_first, node_second_probs[prob_idx] * mean_first
                )
                self._add_edge(
                    node, node_second, node_first_probs[prob_idx] * mean_second
                )

        self._cross_tab.rename(
            {node_first: node_second, node_second: node_first}, axis=0, inplace=True
        )
        self._cross_tab.rename(
            {node_second: node_first, node_first: node_second}, axis=1, inplace=True
        )
        self._recalculate_cross_tab_probabilities()

    def add_node(
        self,
        new_node,
        nodes,
        probabilities,
        mean_time=None,
        time_like=None,
        side="both",
    ):
        """
        Add node and connect it with other nodes to process graph

        Parameters
        ----------
            new_node : str
            nodes : list of str
            probabilities : list of float
                Probability of transition from new_node to target node
            mean_time: TODO
            time_like: TODO
            side : str
                'right', 'left' or 'both'
        """
        if time_like in self._cross_tab.columns:
            self.__time_like_nodes[new_node] = time_like

        self._add_node_to_cross_tab(new_node, mean_time)

        for probability, node in zip(probabilities, nodes):
            self.add_edge(new_node, node, probability, side)

        self._recalculate_cross_tab_probabilities()

    def delete_node(self, node, save_con=True):
        """
        Delete node from graph

        Parameters
        ----------
            node : str
            save_con : bool
                If True, all edges in node, reconnect to all out edges from node
        """
        if save_con:
            self._del_node_create_connections(node)

        self._del_node_from_cross_tab(node)
        self._recalculate_cross_tab_probabilities()

    def add_edge(self, node_first, node_second, prob=0.5, side="right") -> None:
        """
        Add edge, set probability and recompute all probabilities

        Parameters
        ----------
            node_first : str
            node_second : str
            prob : float
                Probability to transition from node to node
            one_side : str
                determines the direction of connection
        """
        # TODO function for each_case
        if side == "right":
            self._add_edge(node_first=node_first, node_second=node_second, prob=prob)
        elif side == "left":
            self._add_edge(node_first=node_second, node_second=node_first, prob=prob)
        else:
            self._add_edge(node_first=node_second, node_second=node_first, prob=prob)
            self._add_edge(node_first=node_first, node_second=node_second, prob=prob)

        self._recalculate_cross_tab_probabilities()

    def delete_edge(self, node_first, node_second, side="right"):
        """
        Delete edge

        Parameters
        ----------
            node_first : str
            node_second : str
            one_side : str
                determines the direction of connection
        """
        self._del_edge(node_first, node_second)

        # TODO function for each_case
        if side == "right":
            self._del_edge(node_first, node_second)
        elif side == "left":
            self._del_edge(node_first=node_second, node_second=node_first)
        else:
            self._del_edge(node_first=node_first, node_second=node_second)
            self._del_edge(node_first=node_second, node_second=node_first)

        self._recalculate_cross_tab_probabilities()

    def delete_loop(self, node):
        """
        Easy delete self cycle edge from process graph

        Parameters
        ----------
            node : str
                node with self cycle
        """
        self.delete_edge(node, node)
        self._recalculate_cross_tab_probabilities()

    def add_loop(self, node, prob=0.5):
        """
        Easy add self cycle edge to process graph

        Parameters
        ----------
            node : str
            prob : float
        """
        self.add_edge(node, node, prob)
        self._recalculate_cross_tab_probabilities()

    def delete_all_loops(self):
        """
        Delete all loops (self cycles) from process graph
        """
        for node in set(self._cross_tab.columns) - {"start", "end"}:
            self._del_edge(node, node)

        self._recalculate_cross_tab_probabilities()

    def change_edge_probability(self, node_first, node_second, new_prob=0):
        """
        Change prob for edge in process graph

        Parameters
        ----------
            node_first : str
                node from edge
            node_second : str
                node to edge
            new_prob : float
                new probability for edge
        """
        self._add_edge(node_first, node_second, new_prob)
        self._recalculate_cross_tab_probabilities()

    def scale_time_node(self, node, scale=1) -> None:
        """
        Change time for activity

        Parameters
        ----------
            node : str
            scale : float
                number
        """
        if node not in self._cross_tab.columns:
            return

        self._action_duration[node] = self._action_duration[node] * scale

    def get_probabilities_tab(self):  # TODO rename tab
        """
        Return cross tab
        """
        return self._cross_tab

    # TODO speed up, no recursion - too many calls plus appending
    # ~10 times number of iterations * 10 - 100 microseconds
    def _rec_choice_trace(self, start_node, trace):
        """
        Recursively constructs a chain of activities.

        Parameters
        ----------
            start_node: str
                Node from which the transition will take place.
            trace: list of str
                Part of the process path to which node are added
        Returns
        -------
            trace: list of str
                Simulated event trace.
        """
        next_node = choice(
            self.__node_dict[start_node][:, 0],
            p=self.__node_dict[start_node][:, 1].astype(float64),
        )

        if next_node == "end":
            return trace

        trace.append(next_node)

        return self._rec_choice_trace(next_node, trace)

    def _add_edge(self, node_first, node_second, prob):
        # TODO function for duplication
        if (
            node_first in self._cross_tab.index
            and node_second in self._cross_tab.columns
        ) or (
            node_first in self._cross_tab.columns
            and node_second in self._cross_tab.index
        ):
            self._cross_tab.at[node_first, node_second] = prob

    def _del_edge(
        self, node_first, node_second
    ):  # ? TODO rename different of delete_edge
        # TODO function for duplication
        if (
            node_first in self._cross_tab.index
            and node_second in self._cross_tab.columns
        ) or (
            node_first in self._cross_tab.columns
            and node_second in self._cross_tab.index
        ):
            self._cross_tab.at[node_first, node_second] = 0

    def _del_node_create_connections(self, node):
        # TODO function for duplication
        if (node in self._cross_tab.columns) and (node not in ["start", "end"]):
            del_node_col = self._cross_tab.loc[node]
            value_node_row = self._cross_tab[node]

            for node_value, tab_index in zip(value_node_row, self._cross_tab.index):
                self._cross_tab.loc[tab_index] = (
                    del_node_col * node_value + self._cross_tab.loc[tab_index]
                ).values

    def _del_node_from_cross_tab(self, node):
        # TODO function for duplication
        if (node in self._cross_tab.columns) and (node not in ["start", "end"]):
            self._cross_tab.drop(node, inplace=True)
            self._cross_tab.drop(columns=[node], inplace=True)

    def _add_node_to_cross_tab(self, node, mean_time):
        if (node not in self._cross_tab.columns) and (node not in ["start", "end"]):
            self._cross_tab.loc[node] = 0
            self._cross_tab[node] = 0

            if mean_time:
                self._action_duration[node] = array([mean_time])

    def _cross_tab_to_dict(self) -> dict:
        """
        Converts crosstab to dict, for use when generating a process

        Returns
        -------
            nodes_to_probs : dict(str : np.array(str, float))
                Dictionary of transitions from node to other nodes with probabilities
        """
        nodes = self._cross_tab.index
        nodes_to_probs: dict[str, dict[str, float64]] = self._cross_tab.T.to_dict()

        return {node: array(tuple(nodes_to_probs[node].items())) for node in nodes}

    def _get_cross_tab(self) -> DataFrame:
        """
        Generate adjacency matrix

        Returns
        -------
            cross_tab : DataFrame
                Adjacency matrix
        """
        unique_actions = list(
            unique(self.data_holder.data[self.data_holder.activity_column])
        ) + [
            "start",
            "end",
        ]

        cross_tab = crosstab(unique_actions, unique_actions) * 0
        mask = self._activity_mutation_data.groupby(
            by=[
                self.data_holder.activity_column,
                self.data_holder.activity_column + "_next",
            ]
        )[self.data_holder.duration_column].count()

        for index in mask.index:
            cross_tab.at[index] = mask[index]

        cross_tab.drop(columns="start", inplace=True)
        cross_tab.drop("end", inplace=True)

        return cross_tab

    def _convert_result_to_df(self):
        """
        Add duration and start_timestamp to generated data, calculates the time for
        each stage of the process and records its date, converts the date format

        Returns
        -------
            result : DataFrame, columns as in init data_holder
        """
        if not self._start_timestamp_traces:
            return ValueError('To start, use the "generate" function.')

        self._result = DataFrame(self._start_timestamp_traces).explode("trace")

        self._result["duration"] = (
            self._result.groupby(self._result.trace)["duration"]
            .transform(
                lambda trace_by_action: self._get_duration_array(
                    trace_by_action.name, len(trace_by_action)
                )
            )
            .values
        )  # TODO check if done

        # TODO speed up
        update_time = self._result.groupby("id_column")[["duration"]].apply(cumsum)

        update_time = update_time.explode("duration")
        self._result["duration_tmp"] = update_time  # TODO refactor

        start_act_mask = self._result.id_column == self._result.id_column.shift(1)

        self._result["tmp"] = self._result.duration_tmp.shift(1)  # TODO refactor

        self._result.loc[start_act_mask, "start"] = self._result[
            start_act_mask
        ].start + to_timedelta(
            self._result[start_act_mask].tmp, unit="s", errors="coerce"
        )

        try:
            self._result.start = self._result.start.dt.strftime(
                self.data_holder.time_format
            )
        except AttributeError:  # FIXME suspicious hardcode
            self._result.start = self._result.start.dt.strftime("%Y-%m-%d %H:%M:%S")

        self._result.rename(
            columns={
                "id_column": self.data_holder.id_column,
                "trace": self.data_holder.activity_column,
                "start": self.data_holder.start_timestamp_column,
                "duration": self.data_holder.duration_column,
            },
            inplace=True,
        )

        # this allows you to generate similar breaks in duration (strongly affects the
        # time series prediction, because in gsp module Nan convert to 0)
        if (
            self.data_holder.start_timestamp_column is not None
            and self.data_holder.end_timestamp_column is not None
        ):
            self._result.drop(
                columns=["tmp", "duration_tmp"], inplace=True
            )  # TODO refactor
        else:
            self._result.drop(
                columns=["tmp", "duration_tmp", "duration"], inplace=True
            )  # TODO refactor

    # TODO refactor
    def change_edges_probabilities(self, probabilities_dict: dict, edges=False):
        """
        Only for timed what-if, non stable
        """
        if not edges:
            cross_tab_nodes = [
                *filter(
                    lambda node: node in self._cross_tab,
                    probabilities_dict,
                )
            ]

            self._cross_tab[cross_tab_nodes] *= tuple(map(abs, probabilities_dict))
        else:
            for first_node in probabilities_dict:
                bound_second_nodes = [
                    *filter(
                        lambda second_node: second_node in self._cross_tab,
                        probabilities_dict[first_node],
                    )
                ]

                self._cross_tab.loc[
                    first_node,
                    bound_second_nodes,
                ] *= tuple(map(abs, probabilities_dict[first_node].values()))

        self._recalculate_cross_tab_probabilities()

    def _recalculate_cross_tab_probabilities(self):
        """
        Recalculates the transition probabilities when the process structure changes.
        """
        if len(self._cross_tab) >= 2:
            self._cross_tab.at["start", "end"] = 0

        for col in self._cross_tab.index:
            self._cross_tab.loc[col] = self._to_prob(self._cross_tab.loc[col].values)

    def _to_prob(self, cross_tabulation_array: NDArray[int64]) -> NDArray[int64]:
        to_prob = np_sum(cross_tabulation_array)

        return (
            cross_tabulation_array / to_prob
            if to_prob
            else zeros_like(cross_tabulation_array)
        )

    def _assign_end_activity_by_process(self) -> DataFrame:
        """
        Adds 'end_event' (and its zero time duration) to the traces in the event log.

        Returns
        -------
            supp_data : pandas.DataFrame
                Modified log data with 'end_event', columns: [activity column (str),
                time duration column (float (minutes)]
        """
        supp_data = (
            self.data_holder.data.groupby(self.data_holder.id_column)
            .agg(
                {
                    self.data_holder.activity_column: tuple,
                    self.data_holder.duration_column: tuple,
                }
            )
            .reset_index()
        )
        supp_data_length = len(supp_data)

        supp_data["act_end"] = [("end",)] * supp_data_length
        supp_data["act_start"] = [("start",)] * supp_data_length
        supp_data["time_end"] = [(0,)] * supp_data_length
        supp_data["time_start"] = [(0,)] * supp_data_length
        supp_data[self.data_holder.activity_column] = (
            supp_data["act_start"]
            + supp_data[self.data_holder.activity_column]
            + supp_data["act_end"]
        )
        supp_data[self.data_holder.duration_column] = (
            supp_data["time_start"]
            + supp_data[self.data_holder.duration_column]
            + supp_data["time_end"]
        )

        supp_data = (
            supp_data[
                [
                    self.data_holder.id_column,
                    self.data_holder.activity_column,
                    self.data_holder.duration_column,
                ]
            ]
            .apply(Series.explode)
            .reset_index(drop=True)
        )
        supp_data[self.data_holder.duration_column] = supp_data[
            self.data_holder.duration_column
        ].fillna(0)
        supp_data[self.data_holder.activity_column + "_next"] = supp_data[
            self.data_holder.activity_column
        ].shift(-1)

        return supp_data

    def _sample_approx_activities_duration(
        self, zeroing_additive=1e-5
    ) -> dict[str, NDArray[float64]]:
        """
        Calculate the approximate and mean_time duration of activities.
        GaussianMixture - function consisting of gaussian functions, the generated distribution +-
        adjusts to the combination of gaussian functions

        Returns
        -------
            acts_duration : dict of {str : numpy.array of float}
                Key: node, value: array of probable time durations.
        """

        def smooth_duration_by_activity(
            duration_by_activity: Series,
        ) -> NDArray[float64]:
            if len(duration_by_activity.value_counts(dropna=True)) > 1:
                return (
                    non_null_smoothed_time(duration_by_activity)
                    if np_sum(duration_by_activity) != 0
                    else all_null_smoothed_time(duration_by_activity)
                )
            else:
                return unique(nan_to_num(duration_by_activity, copy=False, nan=0))

        def non_null_smoothed_time(duration_by_activity: Series) -> NDArray[float64]:
            return array(
                log(
                    duration_by_activity[duration_by_activity != 0] + zeroing_additive
                ).dropna()
            )

        def all_null_smoothed_time(duration_by_activity: Series) -> NDArray[float64]:
            return array(
                log(
                    duration_by_activity
                    + zeroing_additive,  # log(zeroing_additive) of activity_duration size
                ).dropna()
            )

        # TODO rename
        def apply_gaussian_mixture(smoothed_time: NDArray[float64]) -> NDArray[float64]:
            if smoothed_time.size <= 1:
                the_only_time = smoothed_time[0]

                return normal(
                    the_only_time,
                    the_only_time * zeroing_additive,
                    max(smoothed_time.size, 10),
                )

            mixture = GaussianMixture(n_components=min(8, smoothed_time.size)).fit(
                smoothed_time.reshape(-1, 1)
            )
            generated_samples, _ = mixture.sample(n_samples=smoothed_time.size)

            return exp(generated_samples).flatten()

        duration_activity_grouped = self._activity_mutation_data.groupby(
            self.data_holder.activity_column
        )[self.data_holder.duration_column]
        smoothed_time_series = duration_activity_grouped.apply(
            smooth_duration_by_activity
        )

        unique_amount = duration_activity_grouped.nunique(
            dropna=True
        )  # at least two unique non-na durations
        duration_by_activity_series, single_duration_by_activity_series = (
            smoothed_time_series[unique_amount > 1].apply(apply_gaussian_mixture),
            smoothed_time_series[unique_amount <= 1],
        )

        self._mean_duration = mean(duration_by_activity_series.map(mean))

        return {
            **dict(single_duration_by_activity_series),
            **dict(duration_by_activity_series),
        }

    def _get_duration_array(self, node, size, zeroing_additive=1e-5) -> array:
        """
        Generate numpy array duration of action used origin data (if new node +
        like_node='node' -> use duration data from origin data for generate new scaling duration

        Parameters
        ----------
            node : str
                node for which the time is generated
            size : int
                shape for compresses or generate gaussian data time
        Returns
        -------
            duration : np.array
        """
        if node not in self._action_duration:
            return normal(
                loc=self._mean_duration, scale=self._mean_duration**0.5, size=size
            )

        if len(self._action_duration[node]) == 1:
            if node not in self.__time_like_nodes:
                self._action_duration[node] = (
                    normal(1, 0.1, size + 1) * self._action_duration[node]
                )
            else:
                mean_time_scale = (
                    mean(self._action_duration[self.__time_like_nodes[node]])
                    / self._action_duration[node]
                )
                self._action_duration[node] = (
                    self._action_duration[self.__time_like_nodes[node]]
                    * mean_time_scale
                )

        smoothed_dur = log(self._action_duration[node] + 0.1)
        nan_to_num(smoothed_dur, copy=False, nan=0)

        if all(smoothed_dur[0] == smoothed_dur):
            smoothed_dur = normal(smoothed_dur[0], zeroing_additive, smoothed_dur.size)

        kde = gaussian_kde(dataset=smoothed_dur, bw_method="scott")

        return exp(kde.resample(size)[0])

    def _calc_inter_start_rate(self, size: int) -> NDArray[float64]:
        """
        Generates intervals similar to the original between the start of processes,
        compresses the sample to the number of generations (size)

        Parameters
        ----------
            size : int
                shape for compresses data
        Returns
        -------
            start_timestamps: numpy.array of float, shape=[iterations from generate func].
                Time duration in seconds.
        """
        start_timestamp_mask = self.data_holder.data[
            self.data_holder.id_column
        ] != self.data_holder.data[self.data_holder.id_column].shift(1)
        start_timestamps_s_numpy = (
            self.data_holder.data[start_timestamp_mask][
                self.data_holder.start_timestamp_column
            ]
            .dropna()
            .values
        )
        start_timestamps_s = Series(start_timestamps_s_numpy)
        time_periods_s = max(start_timestamps_s) - start_timestamps_s
        time_periods_s.dropna(inplace=True)
        time_periods_s = time_periods_s.apply(lambda x: x.total_seconds())
        kde = gaussian_kde(dataset=log(time_periods_s + 0.1), bw_method="scott")

        return ravel(exp(kde.resample(size)[0]))

    def compute_metric(self, target="activity"):
        """
        Calculates metric (transition, activity or trace), used sberpm functional

        Parameters
        ----------
            target: {'transitions', 'activities', 'trace'}, default='activity'
        Returns
        -------
            transition_metric : TransitionMetric
                OR
            activity_metric : ActivityMetric
                OR
            trace_metric : TraceMetric
        """
        selected_metric = None
        if target == "activity":
            selected_metric = ActivityMetric(self.get_data_holder_result())
        elif target == "transition":
            selected_metric = TransitionMetric(self.get_data_holder_result())
        elif target == "trace":
            selected_metric = TraceMetric(self.get_data_holder_result())

        if selected_metric is not None:
            selected_metric.apply()
            return selected_metric

        return ValueError(
            f'Expected "activity", "transition" or "trace", but got "{target}" instead.'
        )

    def get_data_holder_result(self) -> DataHolder:  # ? TODO better name
        return self._create_holder_like(self.get_result(), self.data_holder)

    def _create_holder_like(self, data: DataFrame, holder_like: DataHolder):
        return DataHolder(
            data=data,
            id_column=holder_like.id_column,
            activity_column=holder_like.activity_column,
            start_timestamp_column=holder_like.start_timestamp_column,
            end_timestamp_column=holder_like.end_timestamp_column,
        )
