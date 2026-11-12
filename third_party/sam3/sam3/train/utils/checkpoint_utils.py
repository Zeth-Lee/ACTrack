# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import contextlib
import fnmatch
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from torch.jit._script import RecursiveScriptModule


def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:

    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        assert len(matching_parameters) > 0, (
            f"param_names {param_name} don't match any param in the given names."
        )
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)


def filter_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:

    if len(patterns) == 0:
        return {}

    all_keys = list(state_dict.keys())
    included_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: state_dict[k] for k in included_keys}


def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:

    if len(patterns) == 0:
        return state_dict

    all_keys = list(state_dict.keys())
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: v for k, v in state_dict.items() if k not in excluded_keys}


def _get_state_dict_summary(state_dict: Dict[str, torch.Tensor]):
    keys = []
    trace = []
    for k, v in state_dict.items():
        keys.append(k)
        trace.append(v.sum().item())
    trace = np.array(trace)[np.argsort(keys)]
    return trace


def assert_skipped_parameters_are_frozen(model: nn.Module, patterns: List[str]):

    if not patterns:
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    non_frozen_keys = {
        n
        for n, p in model.named_parameters()
        if n in frozen_state_dict and p.requires_grad
    }
    if non_frozen_keys:
        raise ValueError(
            f"Parameters excluded with `skip_saving_parameters` should be frozen: {non_frozen_keys}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(
    model: nn.Module, patterns: List[str], disabled: bool = True
):

    if not patterns or disabled:
        yield
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_before = _get_state_dict_summary(frozen_state_dict)

    yield

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_after = _get_state_dict_summary(frozen_state_dict)

    if not np.allclose(summary_before, summary_after, atol=1e-6):
        raise ValueError(
            f"""
            The `model_weight_initializer` has initialized parameters frozen with `skip_saving_parameters`.
            You can resolve this error by either initializing those parameters from within the model definition
            or using the flag `trainer.checkpoint.initialize_after_preemption` to True.
        """
        )


class CkptExcludeKernel:


    def __init__(self, key_pattern: List[str]):
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):

        if len(self.key_pattern) == 0:
            return state_dict
        exclude_keys = unix_pattern_to_parameter_names(
            self.key_pattern, state_dict.keys()
        )
        return {k: v for k, v in state_dict.items() if k not in exclude_keys}


def load_checkpoint(
    path_list: List[str],
    pick_recursive_keys: Optional[List[str]] = None,
    map_location: str = "cpu",
) -> Any:

    path_exists = False
    for path in path_list:
        if g_pathmgr.exists(path):
            path_exists = True
            break

    if not path_exists:
        raise ValueError(f"No path exists in {path_list}")

    with g_pathmgr.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    logging.info(f"Loaded checkpoint from {path}")
    if pick_recursive_keys is not None:
        for key in pick_recursive_keys:
            checkpoint = checkpoint[key]
    return checkpoint


def get_state_dict(checkpoint, ckpt_state_dict_keys):
    if isinstance(checkpoint, RecursiveScriptModule):

        return checkpoint.state_dict()
    pre_train_dict = checkpoint
    for i, key in enumerate(ckpt_state_dict_keys):
        if (isinstance(pre_train_dict, Mapping) and key not in pre_train_dict) or (
            isinstance(pre_train_dict, Sequence) and key >= len(pre_train_dict)
        ):
            key_str = (
                '["' + '"]["'.join(list(map(ckpt_state_dict_keys[:i], str))) + '"]'
            )
            raise KeyError(
                f"'{key}' not found in checkpoint{key_str} "
                f"with keys: {pre_train_dict.keys()}"
            )
        pre_train_dict = pre_train_dict[key]
    return pre_train_dict


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: List[Callable] = None,
    ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    map_location: str = "cpu",
) -> nn.Module:

    assert g_pathmgr.exists(checkpoint_path), "Checkpoint '{}' not found".format(
        checkpoint_path
    )


    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    pre_train_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)


    logging.debug(
        "Loaded Checkpoint State Dict pre-kernel application: %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )

    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            pre_train_dict = f(state_dict=pre_train_dict)

    logging.debug(
        "Loaded Checkpoint State Dict Post-kernel application %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )

    return pre_train_dict


def check_load_state_dict_errors(
    missing_keys,
    unexpected_keys,
    strict: bool,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
):
    if ignore_missing_keys is not None and len(ignore_missing_keys) > 0:
        ignored_keys = unix_pattern_to_parameter_names(
            ignore_missing_keys, missing_keys
        )
        missing_keys = [key for key in missing_keys if key not in ignored_keys]

    if ignore_unexpected_keys is not None and len(ignore_unexpected_keys) > 0:
        ignored_unexpected_keys = unix_pattern_to_parameter_names(
            ignore_unexpected_keys, unexpected_keys
        )
        unexpected_keys = [
            key for key in unexpected_keys if key not in ignored_unexpected_keys
        ]

    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."

    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)


def load_state_dict_into_model(
    state_dict: Dict,
    model: nn.Module,
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    checkpoint_kernels: List[Callable] = None,
):


    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            state_dict = f(state_dict=state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    check_load_state_dict_errors(
        missing_keys,
        unexpected_keys,
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
    )
    return model
