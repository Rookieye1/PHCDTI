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
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import torch
import csv
torch.set_num_threads(20) # 限制torch使用的进程数
# from FuxiCTR_Project import FuxiCTR.fuxictr
from PHCDTI.FuxiCTR.fuxictr import datasets
from datetime import datetime
from PHCDTI.FuxiCTR.fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from PHCDTI.FuxiCTR.fuxictr.features import FeatureMap, FeatureEncoder
from PHCDTI.FuxiCTR.fuxictr.pytorch import models
from PHCDTI.FuxiCTR.fuxictr.pytorch.torch_utils import seed_everything
import gc
import argparse
import logging
import os
from pathlib import Path
# 过滤警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    experments = [
        # "dnn_DrugBank"
        "hoa_DrugBank"
        # "hoa_Davis"
        # "hoa_kiba"
    ]

    for experment_id in experments:
        parser = argparse.ArgumentParser()
        parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
        parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
        parser.add_argument('--config', type=str, default='./config/model',
                            help='The config directory.')
        parser.add_argument('--expid', type=str, default=experment_id, help='The experiment id to run.')
        args = vars(parser.parse_args())
        experiment_id = args['expid']
        params = load_config(args['config'], experiment_id)
        params['gpu'] = args['gpu']
        params['version'] = args['version']
        set_logger(params)
        logging.info(print_to_json(params))
        seed_everything(seed=params['seed'])
        dataset = params['dataset_id'].split('_')[0].lower()
        data_dir = os.path.join(params['data_root'], params['dataset_id'])
        if params.get("data_format") == 'h5': # load data from h5（从h5文件中加载数据集）
            feature_map = FeatureMap(params['dataset_id'], data_dir, params['version'])
            json_file = os.path.join(os.path.join(params['data_root'], params['dataset_id']), "feature_map.json")
            if os.pat\
                    .exists(json_file):
                feature_map.load(json_file)
            else:
                raise RuntimeError('feature_map not exist!')
        else:
            try:
                feature_encoder = getattr(datasets, dataset).FeatureEncoder(**params)
            except:
                feature_encoder = FeatureEncoder(**params)
            if os.path.exists(feature_encoder.json_file):
                feature_encoder.feature_map.load(feature_encoder.json_file)
            else: # Build feature_map and transform h5 data（构建特征映射并转换为h5数据集）
                datasets.build_dataset(feature_encoder, **params)
            params["train_data"] = os.path.join(data_dir, 'train*.h5')
            params["valid_data"] = os.path.join(data_dir, 'valid*.h5')
            params["test_data"] = os.path.join(data_dir, 'test*.h5')
            feature_map = feature_encoder.feature_map
        train_gen, valid_gen = datasets.h5_generator(feature_map, stage='train', **params)
        model_class = getattr(models, params['model'])
        model = model_class(feature_map, **params)
        model.count_parameters()
        model.fit_generator(train_gen, validation_data=valid_gen, **params)
        logging.info("Load best model: {}".format(model.checkpoint))
        model.load_weights(model.checkpoint)
        logging.info('****** Validation evaluation ******')
        valid_result, y_pred, y_true = model.evaluate_generator(valid_gen)
        del train_gen, valid_gen
        logging.info('******** Test evaluation ********')
        test_gen = datasets.h5_generator(feature_map, stage='test', **params)
        if test_gen:
            test_result, y_pred, y_true = model.evaluate_generator(test_gen)
        else:
            test_gen = {}
        result_file = Path(args['config']).name.replace(".yaml", "") + experment_id +'.csv'
        with open(result_file, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                        ' '.join(sys.argv), experiment_id, params['dataset_id'],
                        "N.A.", print_to_list(valid_result)))

        #
        # import pandas as pd
        #
        # print("begin make confidence label")
        # test_data = pd.read_csv(r"../../dataset/criteo/criteo_15g/criteo_train.csv", dtype=str)
        # test_data["pctr"] = [round(i, 6) for i in y_pred]
        # test_data.to_csv(r"../../dataset/multiTask/criteo_with_confidence.csv", index=False)
