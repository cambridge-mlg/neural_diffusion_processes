from typing import Any, Mapping, TypeVar, Type

import dataclasses

from ml_collections import config_dict


DataClass = TypeVar("DataClass")

stdlib_hash = hash


def to_configdict(dc: Any) -> config_dict.ConfigDict:
    return config_dict.ConfigDict(initial_dictionary=dataclasses.asdict(dc))


def to_dataclass(klass: Type[DataClass], data: config_dict.ConfigDict) -> DataClass:
    return _dataclass_from_dict(klass, data)


def get_id(config: DataClass) -> int:
    return abs(stdlib_hash(to_configdict(config).to_json()))


def to_yaml(config: DataClass) -> str:
    return to_configdict(config).__str__()


def _dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: _dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except:
        return d  # Not a dataclass field


def _to_flattened_dict(d: Mapping, prefix: str = "") -> Mapping:
    res = dict()
    for k, v in d.items():
        prefixed_name = f"{prefix}.{k}" if len(prefix) > 0 else k
        if isinstance(v, dict):
            res = {**res, **_to_flattened_dict(v, prefix=prefixed_name)}
        else:
            res[prefixed_name] = v
    return res
