# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import inspect
from functools import wraps
from typing import Callable, TypeVar, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.utils._pytree import tree_map_only


T = TypeVar("T")
Module = TypeVar("Module", bound=nn.Module)


def activation_ckpt_wrapper(module: Union[nn.Module, Callable]) -> Callable:


    @wraps(module)
    def act_ckpt_wrapper(
        *args, act_ckpt_enable: bool = True, use_reentrant: bool = False, **kwargs
    ):
        if act_ckpt_enable:
            if len(args) > 0:
                raise ValueError(
                    "This wrapper expects keyword arguments only when `act_ckpt_enable=True`"
                )

            callable_fn = module.forward if isinstance(module, nn.Module) else module
            sig = inspect.signature(callable_fn)

            param_defaults = {
                name: param.default for name, param in sig.parameters.items()
            }
            args = []
            for p_name in param_defaults.keys():
                if p_name in kwargs:
                    args.append(kwargs.pop(p_name))
                elif param_defaults[p_name] is not inspect.Parameter.empty:

                    args.append(param_defaults[p_name])
                elif (
                    sig.parameters[p_name].kind is not inspect.Parameter.VAR_KEYWORD
                ):
                    raise ValueError(f"Missing positional argument: {p_name}")


            remaining_keys = list(kwargs.keys())
            for key in remaining_keys:
                if isinstance(kwargs[key], torch.Tensor):


                    kwargs[key] = "_REMOVED_BY_ACT_CKPT_WRAPPER_"

            ret = checkpoint.checkpoint(
                module, *args, use_reentrant=use_reentrant, **kwargs
            )
        else:
            ret = module(*args, **kwargs)

        return ret

    return act_ckpt_wrapper


def clone_output_wrapper(f: Callable[..., T]) -> Callable[..., T]:


    @wraps(f)
    def wrapped(*args, **kwargs):
        outputs = f(*args, **kwargs)
        return tree_map_only(
            torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
        )

    return wrapped
