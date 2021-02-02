# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

from glob import glob
import os
import importlib
import pdb
import sys
from ..utils import get_logger

__all__ = ['tasks', 'load_tasks', 'register_task']
tasks={}

logger=get_logger()

def register_task(name=None):
  def register_task_x(cls):
    _name = name
    if _name is None:
      _name = cls.__name__
    _name = _name.lower()
    if _name in tasks:
      logger.warning(f'{_name} already registered in the registry: {tasks[_name]}')
    tasks[_name] = cls
    return cls
  
  if type(name)==type:
    cls = name
    name = None
    return register_task_x(cls)
  return register_task_x

def load_tasks(task_dir = None):
  script_dir = os.path.dirname(os.path.abspath(__file__))
  sys_tasks = glob(os.path.join(script_dir, "tasks/*.py"))
  for t in sys_tasks:
    m = os.path.splitext(os.path.basename(t))[0]
    if not m.startswith('_'):
      importlib.import_module(f'DeBERTa.apps.tasks.{m}')

  if task_dir:
    assert os.path.exists(task_dir), f"{task_dir} must be a valid directory."
    customer_tasks = glob(os.path.join(task_dir, "*.py"))
    sys.path.append(task_dir)
    for t in customer_tasks:
      m = os.path.splitext(os.path.basename(t))[0]
      if not m.startswith('_'):
        importlib.import_module(f'{m}')
