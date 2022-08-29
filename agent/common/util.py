import sys
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import TypeVar
from typing import Union

import attr
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from agent.common.params import Params


def pack_1d_list(sequence: List, out_cls: Type):
    """Pack a list of StepDatas into a BatchData, or a list of SequenceDatas into a BatchSequenceData"""
    out: dict[str, Any] = {}
    for field_obj in attr.fields(type(sequence[0])):
        field = field_obj.name
        if field == "info":
            continue
        sample = getattr(sequence[0], field)
        if isinstance(sample, dict):
            out[field] = {
                k: torch.stack([getattr(transition, field)[k] for transition in sequence]) for k in sample.keys()
            }
        elif np.isscalar(sample):
            out[field] = torch.tensor(np.stack([getattr(transition, field) for transition in sequence]))
        else:
            # tensor
            out[field] = torch.stack([getattr(transition, field) for transition in sequence])
    return out_cls(**out)


def pack_2d_list(batch: List[List], out_cls: Type):
    """Pack a batch of StepDatas into a BatchSequenceData (or subclass)"""
    out: dict[str, Any] = {}
    for k_obj in attr.fields(type(batch[0][0])):
        k = k_obj.name
        if k == "info":
            continue
        example = batch[0][0].__getattribute__(k)
        if np.isscalar(example):
            out[k] = torch.tensor(
                np.stack([np.stack([getattr(transition, k) for transition in trajectory]) for trajectory in batch])
            )
        elif isinstance(example, Tensor):
            out[k] = torch.stack(
                [torch.stack([getattr(transition, k) for transition in trajectory]) for trajectory in batch]
            )
        elif isinstance(example, dict):
            out2: dict[str, Any] = {}
            for k2 in example.keys():
                out2[k2] = torch.stack(
                    [torch.stack([getattr(transition, k)[k2] for transition in trajectory]) for trajectory in batch]
                )
            out[k] = out2
        else:
            assert False
    return out_cls(**out)


def postprocess_uint8_to_float(data: Dict[str, torch.Tensor], observation_prefix: str = ""):
    """Convert uint8 (0,255) to float (-.5, .5) in a dictionary of rollout data.

    We use this to keep images as uint8 in storage + transfer.
    If observation_prefix is passed, then only keys with that prefix will be looked at for transforming.
    This allows handling a whole batch of rollout data (with actions, values, etc), which is convenient.
    """
    out = {}
    for k, v in data.items():
        if k.startswith(observation_prefix) and v.dtype in (np.uint8, torch.uint8):
            v = v / 255.0 - 0.5
        out[k] = v
    return out


ParamType = TypeVar("ParamType", bound=Params)


def strip_optional(field):
    # Optional is an alias for Union[x, None].
    # Return x if this is indeed an Optional, otherwise return field unchanged.
    origin = typing.get_origin(field)
    args = typing.get_args(field)
    if origin is Union and type(None) in args and len(args) == 2:
        assert args[1] == type(None)
        return args[0]
    return field


def parse_args(params: ParamType) -> ParamType:
    """Parse args of the form `test.py --arg1 1 --arg2 value`."""
    args = [x.strip() for x in " ".join(sys.argv[1:]).split("--") if x != ""]
    out = {}
    fields = attr.fields_dict(type(params))
    for arg in args:
        arg, value = arg.split(" ")
        assert arg in fields, f"could not find {arg} in param object."
        dtype = fields[arg].type
        if dtype is None:
            raise ValueError("got none dtype - not sure what this means")
        dtype = strip_optional(dtype)
        if type(dtype) == type:
            if dtype == bool:
                if value.lower() == "false":
                    out[arg] = False
                elif value.lower() == "true":
                    out[arg] = True
                else:
                    raise ValueError(f"could not parse {value} to bool")
            else:
                out[arg] = dtype(value)
        else:
            assert type(dtype) == str, (arg, dtype)
            # We got a forward type declaration, either from an explicit quote-wrapped type, or
            # `from __future__ import annotations`, or by default in future python.
            # In this case dtype will be a string, eg "int" instead of the actual type.
            # In theory attrs.resolve_types or typing.get_type_hints could be used to resolve these, but this breaks
            # if the class has a type defined with a TYPE_CHECKING guard (https://github.com/python/cpython/issues/87629).
            # Using `eval` is one way around this (and suggested in PEP563) but feels unsafe.
            # Probably better to just handle each case manually, eg `if dtype == "str": ...`.
            raise NotImplementedError
    return attr.evolve(params, **out)


ArrayType = Union[NDArray, torch.Tensor]


def explained_variance(y_pred: ArrayType, y_true: ArrayType) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        assert len(y_pred) == len(y_true)
        var_y = torch.var(y_true)  # type: ignore
        return np.nan if var_y == 0 else 1 - torch.var(y_true - y_pred).item() / var_y  # type: ignore
    elif isinstance(y_pred, np.ndarray):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        assert len(y_pred) == len(y_true)
        var_y = np.var(y_true)
        return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred).item() / var_y  # type: ignore
    else:
        import tensorflow as tf

        if tf.is_tensor(y_pred):
            y_pred = tf.reshape(y_pred, [-1])
            y_true = tf.reshape(y_true, [-1])
            assert len(y_pred) == len(y_true)
            var_y = tf.math.reduce_variance(y_true)
            return np.nan if var_y == 0 else 1 - tf.math.reduce_variance(y_true - y_pred).numpy() / var_y
        else:
            raise ValueError
