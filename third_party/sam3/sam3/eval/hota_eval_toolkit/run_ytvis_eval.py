import argparse
import os
import sys
from multiprocessing import freeze_support

from . import trackeval


def run_ytvis_eval(args=None, gt_json=None, dt_json=None):

    default_eval_config = trackeval.Evaluator.get_default_eval_config()

    default_eval_config["PRINT_ONLY_COMBINED"] = True
    default_dataset_config = trackeval.datasets.YouTubeVIS.get_default_dataset_config()
    default_metrics_config = {"METRICS": ["HOTA"]}
    config = {
        **default_eval_config,
        **default_dataset_config,
        **default_metrics_config,
    }
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs="+")
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args(args).__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == "True":
                    x = True
                elif args[setting] == "False":
                    x = False
                else:
                    raise Exception(
                        "Command line parameter " + setting + "must be True or False"
                    )
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {
        k: v for k, v in config.items() if k in default_dataset_config.keys()
    }
    metrics_config = {
        k: v for k, v in config.items() if k in default_metrics_config.keys()
    }


    evaluator = trackeval.Evaluator(eval_config)


    dataset_config["GT_JSON_OBJECT"] = gt_json
    dataset_config["TRACKER_JSON_OBJECT"] = dt_json
    dataset_list = [trackeval.datasets.YouTubeVIS(dataset_config)]
    metrics_list = []


    for metric in [trackeval.metrics.HOTA]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception("No metrics selected for evaluation")
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    return output_res, output_msg


if __name__ == "__main__":
    import sys

    freeze_support()
    run_ytvis_eval(sys.argv[1:])
