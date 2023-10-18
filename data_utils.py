# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import copy
import json
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, TrainingArgumentsAC
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import HfArgumentParser
from data_processer import DataStrategy, TokenIdsMaker
from aigc_zoo.model_zoo.visualglm.llm_model import ChatGLMTokenizer,PetlArguments,ChatGLMConfig,build_masks_and_position_ids_glm
from config import *
from deep_training.nlp.models.visualglm.visual import BlipImageEvalProcessor
from PIL import Image
from io import BytesIO


data_conf = {
   'strategy': DataStrategy.truncation, # 数据策略选项
    DataStrategy.truncation: {
    },
}


def preprocess(text):
  #text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  # return text.replace("\\n", "\n").replace("\\t", "\t")
  return text



class NN_DataHelper(DataHelper):
    index = 1
    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1


        tokenizer: ChatGLMTokenizer
        config: ChatGLMConfig
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        config = self.config

        if not hasattr(self, 'sptoken'):
            self.sptoken = tokenizer.encode(text="")[-2:]

        examples = data

        strategy = data_conf['strategy']
        if strategy == DataStrategy.truncation:
            ds = TokenIdsMaker.tunction(tokenizer,config,examples=examples, max_seq_length=max_seq_length, sptoken=self.sptoken ,**data_conf[strategy])
        else:
            raise ValueError('Invalid strategy',strategy)

        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    def _get_paragraph(self, lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            paragraph = jd['paragraph']
            if line_id < 10:
                print(paragraph)

            prefix = jd.get('p', '')
            paragraph = [(preprocess(session['q']),
                          preprocess('\n'.join(session['a'])) if isinstance(session['a'], list) else preprocess(
                              session['a']))
                         for session in paragraph]
            sub = []
            # 自行做模板
            for (q, a) in paragraph:
                assert len(a), ValueError('answer cannot empty')
                sub.append((q, a))
            D.append((prefix, copy.deepcopy(sub)))
            sub.clear()
        return D

    def _get_messages(self, lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            conversations = jd['conversations']
            if line_id < 10:
                print(conversations)

            paragraph = []
            prefix = ''
            pair = [None, None]
            for m in conversations:
                if m["from"] == 'user':
                    pair[0] = preprocess(m["value"])
                elif m["from"] == 'assistant':
                    pair[1] = preprocess(m["value"])
                elif m["from"] == 'system':
                    prefix = preprocess(m["value"])
                if pair[0] is not None and pair[1] is not None:
                    paragraph.append(tuple(pair))
                    pair[0], pair[1] = None, None

            sub = []
            # 自行做模板
            for (q, a) in paragraph:
                assert len(a), ValueError('answer cannot empty')
                sub.append((q, a))
            D.append((prefix, copy.deepcopy(sub)))
            sub.clear()
        return D

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            is_new = False
            if len(lines) > 0:
                is_new = 'conversations' in json.loads(lines[0])
            if is_new:
                D.extend(self._get_messages(lines))
            else:
                D.extend(self._get_paragraph(lines))
        return D


    def collate_fn(self,batch):
        batch = copy.copy(batch)
        if not hasattr(self,'sptoken'):
            self.sptoken = self.tokenizer.encode(text="")[-2:]

        o = {}
        for i, b in enumerate(batch):
            image_path = b.pop("image_path")
            image_path = image_path[0]
            if isinstance(image_path,bytes):
                image_path = str(image_path, encoding='utf-8')
            if image_path:
                image = Image.open(image_path)
                processor = BlipImageEvalProcessor(224)
                image = processor(image.convert('RGB'))
                b["images"] = image
            else:
                b["pre_image_length"] = torch.zeros(1)

            if i == 0:
                for k in b:
                    value = b[k] if isinstance(b[k], torch.Tensor) else torch.tensor(b[k])
                    o[k] = [value]
            else:
                for k in b:
                    value = b[k] if isinstance(b[k], torch.Tensor) else torch.tensor(b[k])
                    o[k].append(value)
        for k in o:
            o[k] = torch.stack(o[k])


        max_len = torch.max(o.pop('seqlen')).tolist()
        input_ids = o['input_ids'][:, :max_len]
        ctxlens = o.pop('ctxlen')
        assert ctxlens is not None
        # attention_mask,position_ids = build_masks_and_position_ids_glm(input_ids,ctxlens,max_len)
        o['input_ids'] = input_ids.long()
        # o['attention_mask'] = attention_mask.bool()
        # o['position_ids'] = position_ids.long()
        o['labels'] = o['labels'][:, :max_len].long()
        o["pre_image_length"] = torch.max(o["pre_image_length"])
        return o

    def make_dataset_all(self):
        data_args = self.data_args


        # schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "labels": "int32_list",
            "seqlen": "int32_list",
            "ctxlen": "int32_list",
            "image_path": "binary_list",
            "pre_image_length": "int32",
        }
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True,
                                        mode='train', schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)

if __name__ == '__main__':
    if global_args[ "trainer_backend" ] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args,
                                                                            allow_extra_keys=True, )
    elif global_args[ "trainer_backend" ] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments))
        model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)
    elif global_args[ "trainer_backend" ] == "cl":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args, allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005



    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    dataHelper.make_dataset_all()


