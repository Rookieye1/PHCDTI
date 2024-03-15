# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# =========================================================================

import os
import logging
import logging.config
import yaml
import glob
import json
from collections import OrderedDict

# 提取’Base‘信息和experiment_id对应的参数设置，返回字典结构
def load_config(config_dir, experiment_id):
    params = dict()
    # glob.glob() 是一个用于查找文件路径模式匹配的函数。它返回与指定模式匹配的所有文件路径的列表。
    model_configs = glob.glob(os.path.join(config_dir, 'model_config.yaml'))
    if not model_configs:
        # 'model_config/*.yaml'返回config_dir目录下，后缀时.yaml的文件
        model_configs = glob.glob(os.path.join(config_dir, 'model_config/*.yaml'))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    # 此时model_configs是一个包含config_dir路径下的所有后缀为.yaml的列表
    for config in model_configs:
        with open(config, 'r') as cfg:
            # 加载cfg文件中所有信息，返回一个字典
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    if experiment_id not in found_params:
        raise ValueError("expid={} not found in config".format(experiment_id))
    # Update base settings first so that values can be overrided when conflict 
    # with experiment_id settings
    # .update() 方法将传入的字典或可迭代对象中的键值对添加到原始字典中，如果有重复的键，则会更新原始字典中的对应值。
    params.update(found_params.get('Base', {}))
    params.update(found_params.get(experiment_id))
    params['model_id'] = experiment_id
    dataset_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(dataset_params)
    return params


def load_dataset_config(config_dir, dataset_id):
    dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config.yaml'))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config/*.yaml'))
    for config in dataset_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                return config_dict[dataset_id]
    raise RuntimeError('dataset_id={} is not found in config.'.format(dataset_id))


def set_logger(params, log_file=None):
    if log_file is None:
        dataset_id = params['dataset_id']
        model_id = params['model_id']
        log_dir = os.path.join(params['model_root'], dataset_id)
        log_file = os.path.join(log_dir, model_id + '.log')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # logs will not show in the file without the two lines.
    # 清除根日志记录器的处理器，以便重新配置或更改日志记录行为
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
    '''
    用于配置 Python 的 logging 模块，以设置日志记录的级别、格式和处理器。
    level=logging.INFO：设置日志记录的级别为 INFO，这意味着只有 INFO 级别及以上的日志消息才会被记录。
    
    format='%(asctime)s P%(process)d %(levelname)s %(message)s'：设置日志消息的格式。其中，%(asctime)s 表示日志消息的时间戳，P%(process)d 表示进程 ID，
    %(levelname)s 表示日志级别，%(message)s 表示日志消息的内容。
    
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]：设置处理器（handlers）用于将日志消息发送到不同的目标。
    其中，logging.FileHandler(log_file, mode='w') 创建一个文件处理器，将日志消息写入指定的文件（log_file），并使用写入模式（mode='w'）覆盖原有内容。
    logging.StreamHandler() 创建一个流处理器，将日志消息打印到控制台。
    '''
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])

def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())

class Monitor(object):
    def __init__(self, kv):
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pairs = kv

    def get_value(self, logs):
        value = 0
        for k, v in self.kv_pairs.items():
            value += logs.get(k, 0) * v
        return value

def load_h5(data_path, verbose=0):
    if verbose == 0:
        logging.info('Loading data from h5: ' + data_path)
    data_dict = dict()
    with h5py.File(data_path, 'r') as hf:
        for key in hf.keys():
            data_dict[key] = hf[key][:]
    return data_dict