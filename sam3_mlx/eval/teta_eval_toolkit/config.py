"""Config helpers for the TETA compatibility toolkit."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from typing import cast


type ConfigValue = bool | int | str | None | list[str]
type Config = dict[str, ConfigValue]


def _parse_cli(config: Config) -> dict[str, object]:
    parser = argparse.ArgumentParser()
    for setting, value in config.items():
        parser.add_argument(
            "--" + setting,
            nargs="+" if isinstance(value, list) or value is None else None,
        )
    return cast(dict[str, object], vars(parser.parse_args()))


def _apply_cli_overrides(
    config: Config, args: Mapping[str, object], boolean_error: str
) -> None:
    for setting, value in args.items():
        if value is None:
            continue
        current = config[setting]
        if isinstance(current, bool):
            if not isinstance(value, str) or value not in ("True", "False"):
                raise Exception(boolean_error.format(setting=setting))
            config[setting] = value == "True"
        elif isinstance(current, int):
            if not isinstance(value, str):
                raise TypeError(f"Command line parameter {setting} must be an integer")
            config[setting] = int(value)
        else:
            config[setting] = cast(str | list[str], value)


def parse_configs() -> tuple[Config, Config, Config]:
    default_eval_config = get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = True
    default_dataset_config = get_default_dataset_config()
    default_metrics_config: Config = {"METRICS": ["TETA"]}
    config: Config = {
        **default_eval_config,
        **default_dataset_config,
        **default_metrics_config,
    }
    _apply_cli_overrides(
        config,
        _parse_cli(config),
        "Command line parameter {setting} must be True/False",
    )
    return (
        {k: v for k, v in config.items() if k in default_eval_config},
        {k: v for k, v in config.items() if k in default_dataset_config},
        {k: v for k, v in config.items() if k in default_metrics_config},
    )


def get_default_eval_config() -> Config:
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


def get_default_dataset_config() -> Config:
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


def init_config(
    config: Config | None,
    default_config: Mapping[str, ConfigValue],
    name: str | None = None,
) -> Config:
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


def update_config(config: Config) -> Config:
    _apply_cli_overrides(
        config,
        _parse_cli(config),
        "Command line parameter {setting}must be True or False",
    )
    return config


def get_code_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
