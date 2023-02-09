"""
Utilities to monitor and save training runs.

Written by: Erik Bodin (bodin-e)
"""

from abc import abstractmethod
from typing import Optional

import pickle
import typing
from tensorboardX import SummaryWriter
import atexit
import os


DIRECTORY_PATH_VARIABLES = "%s/variables"
DIRECTORY_PATH_OPTIMIZER_STATE = "%s/optimizer_state"
FILENAME_PREFIX = "step_"
FILENAME_SUFFIX = ".pickle"


class Callback:

    @abstractmethod
    def standardized_call(
        self,
        step_index: Optional[int] = None,
        loss=None,
        optimizer_state=None,
        variables=None,
        rng_key=None,
    ):
        pass


class EmptyCallback(Callback):
    def standardized_call(self, **kwargs):
        pass


def call_all(callbacks):
    class Def(Callback):
        def __call__(self, **kwargs):
            for callback in callbacks:
                callback.standardized_call(**kwargs)

        def standardized_call(self, **kwargs):
            self.__call__(**kwargs)

    return Def()


def call_if(callback_fn, filter_fn: typing.Callable[[int], bool]):
    class Def(Callback):
        def __call__(self, step_index, **kwargs):
            if not filter_fn(step_index):
                return
            callback_fn.standardized_call(step_index=step_index, **kwargs)

        def standardized_call(self, **kwargs):
            self.__call__(**kwargs)

    return Def()


def tensorboard_loss_tracker(log_directory, prefix="optimization"):
    writer = SummaryWriter(log_dir=log_directory)

    def close_file():
        writer.close()

    atexit.register(close_file)

    class Def(Callback):
        def __call__(self, step_index, loss):
            if loss is None:
                return
            writer.add_scalar("%s/step_loss" % prefix, loss, step_index)

        def standardized_call(
            self,
            step_index=None,
            loss=None,
            optimizer_state=None,
            variables=None,
            rng_key=None,
        ):
            self.__call__(step_index=step_index, loss=loss)

    return Def()


def checkpoint_saver(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    class Def(Callback):
        def __call__(self, step_index, optimizer_state, variables):
            save_checkpoint(
                optimizer_state=optimizer_state,
                variables=variables,
                directory_path=directory,
                step_index=step_index,
            )

        def standardized_call(
            self,
            step_index=None,
            loss=None,
            optimizer_state=None,
            variables=None,
            rng_key=None,
        ):
            self.__call__(
                step_index=step_index,
                optimizer_state=optimizer_state,
                variables=variables,
            )

    return Def()


def save_pytree(pytree, directory_path: str, step_index: int):
    _cond_mkdir(directory_path)
    filepath = directory_path + "/" + _index_to_checkpoint_filename(step_index=step_index)
    with open(filepath, "wb") as file:
        pickle.dump(pytree, file)


def load_pytree(pytree_unloaded, directory_path: str, step_index: int):
    filepath = directory_path + "/" + _index_to_checkpoint_filename(step_index=step_index)
    with open(filepath, "rb") as file:
        pytree = filepath.load(file)
    return pytree


def save_checkpoint(optimizer_state, variables, directory_path: str, step_index: int):
    assert isinstance(step_index, int)
    _cond_mkdir(directory_path)
    save_pytree(
        optimizer_state,
        directory_path=DIRECTORY_PATH_OPTIMIZER_STATE % directory_path,
        step_index=step_index,
    )
    save_pytree(
        variables,
        directory_path=DIRECTORY_PATH_VARIABLES % directory_path,
        step_index=step_index,
    )


def load_checkpoint(optimizer_state, variables, directory_path: str, step_index: int):
    return (
        load_pytree(
            optimizer_state,
            directory_path=DIRECTORY_PATH_OPTIMIZER_STATE % directory_path,
            step_index=step_index,
        ),
        load_pytree(
            variables,
            directory_path=DIRECTORY_PATH_VARIABLES % directory_path,
            step_index=step_index,
        ),
    )


def load_variables(directory_path, variables, step_index):
    return load_pytree(
        variables,
        directory_path=DIRECTORY_PATH_VARIABLES % directory_path,
        step_index=step_index,
    )


def find_latest_checkpoint_step_index(directory_path: str):
    path = DIRECTORY_PATH_VARIABLES % directory_path
    if not os.path.exists(path):
        return None
    indices = [
        _checkpoint_filename_to_index(name)
        for name in os.listdir(path)
        if name.endswith(FILENAME_SUFFIX)
    ]
    if len(indices) == 0:
        return None
    return max(indices)


def _index_to_checkpoint_filename(step_index: int):
    return "%s%s%s" % (FILENAME_PREFIX, step_index, FILENAME_SUFFIX)


def _checkpoint_filename_to_index(filename: str):
    start = filename.index(FILENAME_PREFIX) + len(FILENAME_PREFIX)
    end = filename.index(FILENAME_SUFFIX)
    filename_part = filename[start:end]
    return int(filename_part)


def _cond_mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)
