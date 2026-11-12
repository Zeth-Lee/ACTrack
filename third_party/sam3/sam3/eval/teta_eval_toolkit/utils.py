import csv
import os
from collections import OrderedDict


def validate_metrics_list(metrics_list):

    metric_names = [metric.get_name() for metric in metrics_list]

    if len(metric_names) != len(set(metric_names)):
        raise TrackEvalException(
            "Code being run with multiple metrics of the same name"
        )
    fields = []
    for m in metrics_list:
        fields += m.fields

    if len(fields) != len(set(fields)):
        raise TrackEvalException(
            "Code being run with multiple metrics with fields of the same name"
        )
    return metric_names


def get_track_id_str(ann):

    if "track_id" in ann:
        tk_str = "track_id"
    elif "instance_id" in ann:
        tk_str = "instance_id"
    elif "scalabel_id" in ann:
        tk_str = "scalabel_id"
    else:
        assert False, "No track/instance ID."
    return tk_str


class TrackEvalException(Exception):


    ...
