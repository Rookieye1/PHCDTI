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
    # 利用实验id组来进行多个模型在多个数据集上的实验（真正的参数定义见‘./config/ctrmodel_in_criteo&avazu/mode l_config.yaml'）
    experments = [
        # "widedeep_taobao"
        # "fibinet_taobao"
        #  "afm_taobao"
        #  "dnn_taobao"
        #  "autoint_taobao"
        #  "ffm_taobao"
        #  "xdeepfm_taobao"
        #  "dcnv2_taobao"
        # "dadc_taobao"
        # "dadc_test_taobao"
        # "onn_taobao"
        # "dadc_test_taobao"

        # "widedeep_movielens"
        #  "fibinet_movielens"
        #   "afm_movielens"
        #  "dnn_movielens"
        #  "autoint_movielens"
        #  "ffm_movielens"
        #  "xdeepfm_movielens"
        #  "dcnv2_movielens"
        # "dadc_movielens"
        # "dadc_tiny"
        # "onn_movielens
        # "dadc_test_movielens"


        # "dnn_DrugBank"
        "hoa_DrugBank"
        # "hoa_Davis"
        # "hoa_kiba"

        # "hoa_DrugBank"
        # "autoint_DrugBank"
        # "fibinet_DrugBank"
        # "xdeepfm_DrugBank"

        # "dnn_knn_test"

    ]

    for experment_id in experments:
        '''当调用parser.print_help()
        或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()
        方法)时，会打印这些描述信息'''
        #定义程序所期望的参数和选项，并自动生成帮助信息和错误处理
        parser = argparse.ArgumentParser()
        # 指定模型依赖框架、运行设备
        # name or flags 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
        # action 当参数在命令行中出现时使用的动作基本类型。
        # nargs 命令行参数应当消耗的数目。
        # const 被一些action和 nargs选择所需求的常数。
        # default 当参数未在命令行中出现并且也不存在于命名空间对象时所产生的值。
        # type 命令行参数应当被转换成的类型。
        # choices 可用的参数的容器。
        # required 此命令行选项是否可省略 （仅选项可用）。
        # help 一个此选项作用的简单描述。
        # metavar 在使用方法消息中使用的参数值示例。
        # dest 被添加到 [parse_args()] 所返回对象上的属性名。
        parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
        parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
        # 指定config目录文件夹所在
        parser.add_argument('--config', type=str, default='./config/ctrmodel_in_criteo&avazu',
                            help='The config directory.')
        # 指定model_config.yaml中的实验id（以此来控制实验）
        parser.add_argument('--expid', type=str, default=experment_id, help='The experiment id to run.')

        # 将上述相关参数写入json文件，并保存，.parse_args()将参数转化为适当类型后操作，vars()函数返回对象object的属性和属性值的字典对象
        # args所包含的具体内容{'version': 'pytorch', 'gpu': 0, 'config': './config/ctrmodel_in_criteo&avazu', 'expid': 'hofm_Criteo'}
        args = vars(parser.parse_args())
        experiment_id = args['expid']
        # 导入model_config.yaml的base以及使用的具体的模型信息(experiment_id参数所决定)和训练模型所采用数据的信息(dataset_config.yaml文件)，字典变量
        params = load_config(args['config'], experiment_id)
        params['gpu'] = args['gpu']
        params['version'] = args['version']
        # set_logger(params) 用于根据 params 字典中提供的参数配置日志记录器对象
        set_logger(params)
        logging.info(print_to_json(params))
        # 设置随机数的用处是在用随机梯度下降、dropout等算法时可重复
        seed_everything(seed=params['seed'])

        # preporcess the dataset（预处理数据集）
        #取得数据集的小写名称
        dataset = params['dataset_id'].split('_')[0].lower()
        #拼接路径的方法
        data_dir = os.path.join(params['data_root'], params['dataset_id'])
        if params.get("data_format") == 'h5': # load data from h5（从h5文件中加载数据集）
            feature_map = FeatureMap(params['dataset_id'], data_dir, params['version'])
            json_file = os.path.join(os.path.join(params['data_root'], params['dataset_id']), "feature_map.json")
            if os.pat\
                    .exists(json_file):
                feature_map.load(json_file)
            else:
                raise RuntimeError('feature_map not exist!')
        else: # load data from csv（如果首次读入的是CSV文件，则进行h5文件的构建）
            try:
                # **params中包含了所有信息的字典
                feature_encoder = getattr(datasets, dataset).FeatureEncoder(**params)
                # 获取datasets模块的dataset属性，也就是本次实验中fuxictr.datasets的criteo.py。函数返回一个FeatureEncoder类对象。
                # 这个类对象可以用来创建feature_encoder对象，用于对数据进行编码。
            except:
                feature_encoder = FeatureEncoder(**params)
                # FeatureEncoder中self.json_file = os.path.join(self.data_dir, "feature_map.json")
            if os.path.exists(feature_encoder.json_file):
                feature_encoder.feature_map.load(feature_encoder.json_file)
            else: # Build feature_map and transform h5 data（构建特征映射并转换为h5数据集）
                datasets.build_dataset(feature_encoder, **params)
            # 用于路径拼接文件路径，可以传入多个路径
            params["train_data"] = os.path.join(data_dir, 'train*.h5')
            params["valid_data"] = os.path.join(data_dir, 'valid*.h5')
            params["test_data"] = os.path.join(data_dir, 'test*.h5')
            # 输出稀疏特征的特征的字典
            feature_map = feature_encoder.feature_map
        # get train and validation data（获取训练与验证数据集，均来源于训练集）
        # ** params中包含了字典params中所有的键值对，包括指定数据集的路径、批次大小、数据增强方式等
        # 例如params = { 'data_path': 'path/to/data.h5','batch_size': 32,'augmentation': True,'shuffle': True}
        train_gen, valid_gen = datasets.h5_generator(feature_map, stage='train', **params)

        # initialize model（初始化模型，并将相关参数设置归于模型中，
        # models为Fuxictr目录下的models,params['model']为选用的模型名称
        model_class = getattr(models, params['model'])
        #根据模型参数，使用model_class生成模型对象model，并输出模型参数数量
        model = model_class(feature_map, **params)
        # print number of parameters used in model（打印模型总训练参数量）
        model.count_parameters()
        # fit the model（拟合模型）
        model.fit_generator(train_gen, validation_data=valid_gen, **params)

        # load the best model checkpoint（载入最佳模型参数）
        #model.checkpoint为C:\Users\yexia\PycharmProjects\pythonProject\FuxiCTR\checkpoints\criteo\deepfm_Criteo.model，记录日志信息
        logging.info("Load best model: {}".format(model.checkpoint))
        # model.load_weights(model.checkpoint)保存在model.checkpoint文件中的模型参数加载到当前模型中。恢复模型的训练状态，可以继续训练模型或使用该模型进行预测
        model.load_weights(model.checkpoint)

        # get evaluation results on validation（获取模型在验证集上的结果）
        logging.info('****** Validation evaluation ******')
        valid_result, y_pred, y_true = model.evaluate_generator(valid_gen)
        del train_gen, valid_gen
        # gc.collect()
        # get evaluation results on test（获取模型在测试集上的结果）
        logging.info('******** Test evaluation ********')
        test_gen = datasets.h5_generator(feature_map, stage='test', **params)
        if test_gen:
            test_result, y_pred, y_true = model.evaluate_generator(test_gen)
            # print(y_pred)
            # #创建文件名
            # output_file_name = experment_id + '_result.csv'
            # with open(output_file_name, 'w', newline='') as csvfile:
            #     # 创建csv写入器,writer是一个类
            #     writer = csv.writer(csvfile)
            #     # 写入表头
            #     writer.writerow(['y_pred', 'y_true'])
            #     for i in range(len(y_pred)):
            #         writer.writerow([y_pred.tolist()[i], y_true.tolist()[i]])
            # # print(y_pred.tolist(), y_true.tolist())
        else:
            test_gen = {}
        # save the results to csv（将当前批次运行的结果保存至csv文件中）
        # result_file:ctrmodel_in_criteo&avazu.csv; Path(args['config']):config\ctrmodel_in_criteo&avazu
        # Path(args['config']).name:ctrmodel_in_criteo&avazu
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