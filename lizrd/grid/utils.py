import argparse
import copy
import pprint
from itertools import product
from typing import List, Tuple
import sys


def split_params(params: dict) -> Tuple[list, list, list]:
    functions = []
    grids = []
    normals = []
    for k, v in params.items():
        if k[0] == "^":
            grids.append((k[1:], v))
        elif k[0] == "*":
            functions.append((k[1:], v))
        elif "," in k:
            grids.append((k, v))
        elif isinstance(v, dict):
            sub_grid = create_grid(v)
            grids.append((k, sub_grid))
        else:
            normals.append((k, v))
    return grids, functions, normals


def shorten_arg(arg: str) -> str:
    ARG_TO_ABBR = {
        "reinit_dist": "rd",
        "ff_layer": "ff",
        "mask_loss_weight": "mlw",
        "class_loss_weight": "clw",
        "mask_percent": "mp",
        "n_steps": "ns",
        "n_steps_eval": "nse",
        "immunity": "im",
        "pruner": "pr",
        "pruner_prob": "prp",
        "pruner_delay": "prd",
        "pruner_n_steps": "prns",
    }
    return ARG_TO_ABBR.get(arg, arg)


def shorten_val(val: str) -> str:
    VAL_TO_ABBR = {
        # ff layers
        "regular": "r",
        "unstruct_prune": "up",
        "struct_prune": "sp",
        "unstruct_magnitude_prune": "ump",
        "struct_magnitude_prune": "smp",
        "unstruct_magnitude_recycle": "umr",
        "struct_magnitude_recycle_with_immunity": "smri",
        "masked_ff": "mf",
        "separate_direction_magnitude_ff": "sdmf",
        # reinit dist
        "zero": "0",
        "init": "i",
        "follow_normal": "fn",
    }
    if isinstance(val, bool):
        return "T" if val else "F"
    if isinstance(val, str):
        return VAL_TO_ABBR.get(val, val)
    if isinstance(val, int):
        if val % 1_000_000 == 0:
            return f"{val // 1_000_000}M"
        if val % 1_000 == 0:
            return f"{val // 1_000}k"
        return str(val)

    return str(val)


def make_tags(arg, val) -> str:
    if isinstance(val, list):
        val = "_".join([shorten_val(v) for v in val])
    if isinstance(val, dict) and "tags" in val:
        to_return = val["tags"]
        del val["tags"]
        to_return = [f"{shorten_arg(arg)}:{tag}" for tag in to_return]
        return to_return
    return f"{shorten_arg(arg)}={shorten_val(val)}"


# parse time to minutes
def timestr_to_minutes(time: str) -> int:
    # Supported formats: "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
    def parse_time_part_no_days(time_part: str) -> Tuple[int, int, int]:
        if sum(c == ":" for c in time_part) == 0:
            return 0, int(time_part), 0
        elif sum(c == ":" for c in time_part) == 1:
            minutes, seconds = time_part.split(":")
            return 0, int(minutes), int(seconds)
        elif sum(c == ":" for c in time_part) == 2:
            hours, minutes, seconds = time_part.split(":")
            return int(hours), int(minutes), int(seconds)
        else:
            raise ValueError(f"Invalid time format: {time_part}")

    def parse_time_part_with_days(time_part: str) -> Tuple[int, int, int]:
        if sum(c == ":" for c in time_part) == 0:
            return int(time_part), 0, 0
        elif sum(c == ":" for c in time_part) == 1:
            hours, minutes = time_part.split(":")
            return int(hours), int(minutes), 0
        elif sum(c == ":" for c in time_part) == 2:
            hours, minutes, seconds = time_part.split(":")
            return int(hours), int(minutes), int(seconds)
        else:
            raise ValueError(f"Invalid time format: {time_part}")

    if "-" in time:
        days_part, time_part = time.split("-")
        days = int(days_part)
        hours, minutes, seconds = parse_time_part_with_days(time_part)
    else:
        days = 0
        hours, minutes, seconds = parse_time_part_no_days(time)

    return days * 24 * 60 + hours * 60 + minutes + round(seconds / 60)


def seconds_to_timestr(seconds: int) -> str:
    """return format hh:mm:ss"""
    return f"{int(seconds/(60*60))}:{int((seconds%(60*60))/60)}:{int(seconds%60)}"


def create_grid(params: dict) -> List[dict]:
    grids, functions, normals = split_params(params)
    base_params = {k: v for k, v in normals}
    out_params = []
    grids_keys = [k for k, v in grids]
    grids_values = product(*(v for k, v in grids))
    for value in grids_values:
        out_dict = copy.deepcopy(base_params)
        value = copy.deepcopy(
            value
        )  # this is grids over dictionaries, we need to delete the tags inside
        grid_dict = dict(zip(grids_keys, value))
        tags = []
        for k, v in grid_dict.items():
            new_tags = make_tags(k, v)
            if isinstance(new_tags, list):
                tags.extend(new_tags)
            else:
                tags.append(new_tags)
        if out_dict.get("tags") is None:
            out_dict["tags"] = []
        out_dict["tags"].extend(tags)
        out_dict = {**out_dict, **grid_dict}
        for func_name, func in functions:
            out_dict[func_name] = func(out_dict)
        out_params.append(out_dict)

    return out_params


def multiply_grid(param_sets: List[dict], runs_count: int) -> List[dict]:
    assert runs_count > 0

    if runs_count == 1:
        return param_sets

    out_params_sets = []
    for param_set in param_sets:
        for i in range(runs_count):
            out_dict = copy.deepcopy(param_set)
            out_dict["tags"].append(f"run={i+1}")
            out_dict["tags"].append(f"num_runs={runs_count}")
            out_params_sets.append(out_dict)
    return out_params_sets


def unpack_params(k, v):
    if "," in k:
        k = k.split(",")
        return k, v
    return [k], [v]


def param_to_str(param) -> str:
    if isinstance(param, str):
        return " ".join(param)
    else:
        return str(param)


def list_to_clean_str(l: List[str]) -> str:
    return " ".join([str(s) for s in l if s is not None])


def get_train_main_function(runner: str):
    __import__(runner)
    runner_module = sys.modules.get(runner, None)
    if runner_module is None:
        raise ValueError(f"Unknown runner: {runner}")
    return runner_module.main


def translate_to_argparse(param_set: dict):
    runner_params = []

    for k_packed, v_packed in param_set.items():
        for k, v in zip(*unpack_params(k_packed, v_packed)):
            if isinstance(v, bool):
                if v:
                    runner_params.append(f"--{k}")
                else:
                    pass  # simply don't add it if v == False
                continue
            elif v is not None:  # None values should not be added
                runner_params.append(f"--{k}")
                if isinstance(v, list):
                    runner_params.extend([str(s) for s in v])
                elif isinstance(v, dict):
                    runner_params.append(f'{str(v).replace(" ", "")}')
                else:
                    runner_params.append(str(v))

    return runner_params


def check_for_argparse_correctness(grid: list[dict[str, str]]):
    for setup_args, trainings_args in grid:
        for i, training_args in enumerate(trainings_args):
            runner_params = translate_to_argparse(training_args)
            runner = setup_args["runner"]
            argparse_module_name = setup_args.get("argparse", None)
            if argparse_module_name is None:
                if "/" in runner:
                    argparse_module_name = runner.split("/")
                else:
                    argparse_module_name = runner.split(".")
                argparse_module_name[-1] = "argparse"
                argparse_module_name[-2] = "utils"
                argparse_module_name = ".".join(argparse_module_name)
            __import__(argparse_module_name)
            argparse_module = sys.modules.get(argparse_module_name, None)
            if argparse_module is None:
                raise ValueError(f"Unknown argparse module: {argparse_module_name}")

            parser = argparse.ArgumentParser()
            parser = argparse_module.introduce_parser_arguments(parser)

            try:
                args, extra = parser.parse_known_args(runner_params)
                if extra != []:
                    print("Config:")
                    pprint.pprint(runner_params)
                    raise ValueError(f"Unknown arguments: {extra}")

            except SystemExit as e:
                print(f"Error in config {i}: {e}")
                print("Config:")
                pprint.pprint(runner_params)
                raise e


def setup_experiments(configs: dict):
    grid = []
    for infra_config, training_config in configs:
        pprint.pprint(training_config)
        single_exp_training_args_grid = create_grid(training_config)
        multiplied_grid = multiply_grid(
            single_exp_training_args_grid, infra_config["runs_multiplier"]
        )
        grid.append((infra_config, multiplied_grid))
    return grid
