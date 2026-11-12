# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import atexit
import functools
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Union

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from sam3.train.utils.train_utils import get_machine_local_and_dist_rank, makedir
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

Scalar = Union[Tensor, ndarray, int, float]


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)
    summary_writer_method = SummaryWriter
    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


class TensorBoardWriterWrapper:


    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:

        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:


        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:


        if not self._writer:
            return

        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):


    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:

        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:

        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:

        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)


class Logger:


    def __init__(self, logging_conf):

        tb_config = logging_conf.tensorboard_writer
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log(name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):


    log_buffer_kb = 10 * 1024
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):


    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)


    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)


    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []


    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)


    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging():

    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()
