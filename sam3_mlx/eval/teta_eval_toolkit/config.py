"""Config helpers for the TETA compatibility toolkit."""

from __future__ import annotations

import argparse
import os


def parse_configs():
    default_eval_config = get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = True
    default_dataset_config = get_default_dataset_config()
    default_metrics_config = {"METRICS": ["TETA"]}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}
    parser = argparse.ArgumentParser()
    for setting, value in config.items():
        parser.add_argument(
            "--" + setting,
            nargs="+" if isinstance(value, list) or value is None else None,
        )
    args = parser.parse_args().__dict__
    for setting, value in args.items():
        if value is not None:
            if isinstance(config[setting], bool):
                if value not in ("True", "False"):
                    raise Exception(
                        f"Command line parameter {setting} must be True/False"
                    )
                value = value == "True"
            elif isinstance(config[setting], int):
                value = int(value)
            config[setting] = value
    return (
        {k: v for k, v in config.items() if k in default_eval_config},
        {k: v for k, v in config.items() if k in default_dataset_config},
        {k: v for k, v in config.items() if k in default_metrics_config},
    )


def get_default_eval_config():
    code_path = get_code_path()
    return {
        "USE_PARALLEL": True,
        "NUM_PARALLEL_CORES": 8,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "LOG_ON_ERROR": os.path.join(code_path, "error_log.txt"),
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": True,
        "TIME_PROGRESS": True,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "OUTPUT_TEM_RAW_DATA": True,
        "OUTPUT_PER_SEQ_RES": True,
    }


def get_default_dataset_config():
    code_path = get_code_path()
    return {
        "GT_FOLDER": os.path.join(code_path, "data/gt/tao/tao_training"),
        "TRACKERS_FOLDER": os.path.join(code_path, "data/trackers/tao/tao_training"),
        "OUTPUT_FOLDER": None,
        "TRACKERS_TO_EVAL": ["TETer"],
        "CLASSES_TO_EVAL": None,
        "SPLIT_TO_EVAL": "training",
        "PRINT_CONFIG": True,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "MAX_DETECTIONS": 0,
        "USE_MASK": False,
    }


def init_config(config, default_config, name=None):
    if config is None:
        config = dict(default_config)
    else:
        for key, value in default_config.items():
            config.setdefault(key, value)
    if name and config["PRINT_CONFIG"]:
        print("\n%s Config:" % name)
        for key in config:
            print("%-20s : %-30s" % (key, config[key]))
    return config


def update_config(config):
    parser = argparse.ArgumentParser()
    for setting, value in config.items():
        parser.add_argument(
            "--" + setting,
            nargs="+" if isinstance(value, list) or value is None else None,
        )
    args = parser.parse_args().__dict__
    for setting, value in args.items():
        if value is not None:
            if isinstance(config[setting], bool):
                if value not in ("True", "False"):
                    raise Exception(
                        "Command line parameter " + setting + "must be True or False"
                    )
                value = value == "True"
            elif isinstance(config[setting], int):
                value = int(value)
            config[setting] = value
    return config


def get_code_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
