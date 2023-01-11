# Copyright 2022 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Logger."""

import atexit
import json
import os
import os.path as osp
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from omnisafe.utils.distributed_utils import mpi_statistics_scalar, proc_id
from omnisafe.utils.logger_utils import colorize, convert_json


# pylint: disable-next=too-many-instance-attributes
class Logger:
    """Implementation of the Logger."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        data_dir,
        exp_name,
        output_fname='progress.txt',
        debug=False,
        level=1,
        datestamp=True,
        hms_time=time.strftime('%Y-%m-%d_%H-%M-%S'),
        use_tensor_board=True,
        verbose=True,
        seed=None,
    ):
        """Initialize the logger."""
        relpath = hms_time if datestamp else ''
        if seed is not None:
            subfolder = '-'.join(['seed', str(seed).zfill(3)])
            relpath = '-'.join([subfolder, relpath])
        relpath += str(os.getpid())
        self.log_dir = os.path.join(data_dir, exp_name, relpath)
        self.debug = debug if proc_id() == 0 else False
        self.level = level
        # only the MPI root process is allowed to print information to console
        self.verbose = verbose if proc_id() == 0 else False

        if proc_id() == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            # pylint: disable-next=consider-using-with
            self.output_file = open(
                osp.join(self.log_dir, output_fname), encoding='utf-8', mode='w'
            )
            atexit.register(self.output_file.close)
            print(colorize(f'Logging data to {self.output_file.name}', 'cyan', bold=True))
        else:
            self.output_file = None

        self.epoch = 0
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.torch_saver_elements = None

        # Setup tensor board logging if enabled and MPI root process
        self.summary_writer = (
            SummaryWriter(os.path.join(self.log_dir, 'tb'))
            if use_tensor_board and proc_id() == 0
            else None
        )

        self.epoch_dict = {}

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if self.verbose and self.level > 0:
            print(colorize(msg, color, bold=False))

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for key, value in kwargs.items():
            if key not in self.epoch_dict:
                self.epoch_dict[key] = []
            self.epoch_dict[key].append(value)

    def log_single_value(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert (
                key in self.log_headers
            ), f"Trying to introduce a new key {key} that you didn't include in the first iteration"
        assert (
            key not in self.log_current_row
        ), f'You already set {key} this iteration. Maybe you forgot to call dump_tabular()'
        self.log_current_row[key] = val

    def log_tabular(self, key, val=None, min_and_max=False, std=False):
        """log_tabular"""

        if val is not None:
            self.log_single_value(key, val)
        else:
            stats = self.get_stats(key, min_and_max)
            if min_and_max or std:
                self.log_single_value(key + '/Mean', stats[0])
            else:
                self.log_single_value(key, stats[0])
            if std:
                self.log_single_value(key + '/Std', stats[1])
            if min_and_max:
                self.log_single_value(key + '/Min', stats[2])
                self.log_single_value(key + '/Max', stats[3])
        self.epoch_dict[key] = []

    def save_config(self, config):
        """Save the configuration of the experiment."""

        if proc_id() == 0:  # only root process logs configurations
            config_json = convert_json(config)
            output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
            # if self.verbose and self.level > 0:
            #     print(colorize('Run with config:', color='yellow', bold=True))
            #     print(output)
            print(colorize('Save with config in config.json', color='yellow', bold=True))
            with open(osp.join(self.log_dir, 'config.json'), encoding='utf-8', mode='w') as out:
                out.write(output)

    def get_stats(self, key, with_min_and_max=False):
        """Get the statistics of a key."""
        assert key in self.epoch_dict, f'key={key} not in dict'

        val = self.epoch_dict[key]
        vals = (
            np.concatenate(val) if isinstance(val[0], np.ndarray) and len(val[0].shape) > 0 else val
        )
        return mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)

    def dump_tabular(self) -> None:
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if proc_id() == 0:
            vals = []
            self.epoch += 1
            # Print formatted information into console
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%' + '%d' % max_key_len  # pylint: disable=consider-using-f-string
            fmt = '| ' + keystr + 's | %15s |'
            n_slashes = 22 + max_key_len
            if self.verbose and self.level > 0:
                print('-' * n_slashes)
            # print('-' * n_slashes) if self.verbose and self.level > 0 else None
            for key in self.log_headers:
                val = self.log_current_row.get(key, '')
                # pylint: disable-next=consider-using-f-string
                valstr = '%8.3g' % val if hasattr(val, '__float__') else val
                if self.verbose and self.level > 0:
                    print(fmt % (key, valstr))
                vals.append(val)
            if self.verbose and self.level > 0:
                print('-' * n_slashes, flush=True)

            # Write into the output file (can be any text file format, e.g. CSV)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write(' '.join(self.log_headers) + '\n')
                self.output_file.write(' '.join(map(str, vals)) + '\n')
                self.output_file.flush()

            if self.summary_writer is not None:
                for key, value in zip(self.log_headers, vals):
                    self.summary_writer.add_scalar(key, value, global_step=self.epoch)

                # Flushes the event file to disk. Call this method to make sure
                # that all pending events have been written to disk.
                self.summary_writer.flush()

        # free logged information in all processes...
        self.log_current_row.clear()
        self.first_row = False

        # Check if all values from dict are dumped -> prevent memory overflow
        for key, value in self.epoch_dict.items():
            if len(value) > 0:
                print(f'epoch_dict: key={key} was not logged.')

    def setup_torch_saver(self, what_to_save: dict):
        """Setup the torch saver."""
        self.torch_saver_elements = what_to_save

    def torch_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        if proc_id() == 0:
            assert (
                self.torch_saver_elements is not None
            ), 'First have to setup saving with self.setup_torch_saver'
            fpath = 'torch_save'
            fpath = osp.join(self.log_dir, fpath)
            fname = f'model {itr}.pt'
            fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)

            params = {
                k: v.state_dict() if isinstance(v, torch.nn.Module) else v
                for k, v in self.torch_saver_elements.items()
            }
            torch.save(params, fname)

    def close(self):
        """
        Close opened output files immediately after training in order to
        avoid number of open files overflow. Avoids the following error:
        OSError: [Errno 24] Too many open files
        """
        if proc_id() == 0:
            self.output_file.close()
