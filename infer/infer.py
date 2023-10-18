# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper
from aigc_zoo.model_zoo.visualglm.llm_model import MyTransformer,ChatGLMTokenizer,setup_model_profile, ChatGLMConfig
from aigc_zoo.model_zoo.visualglm.llm_model import RotaryNtkScaledArguments,RotaryLinearScaledArguments # aigc-zoo 0.1.21

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args,allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args)
    tokenizer: ChatGLMTokenizer
    config: ChatGLMConfig
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)
    assert tokenizer.eos_token_id == 130005
    config.initializer_weight = False

    rope_args = None
    enable_rope = False
    if enable_rope:
        rope_args = RotaryNtkScaledArguments(model_type='chatglm',max_position_embeddings=config.max_sequence_length,alpha=4) # 扩展 8k
    # rope_args = RotaryLinearScaledArguments(model_type='chatglm',name='rotary_pos_emb',max_position_embeddings=2048, scale=4) # 扩展 8k
    
    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16,rope_args=rope_args)

    model = pl_model.get_llm_model()
    if not model.quantized:
        # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
        model.half().quantize(4).cuda()
    else:
        # 已经量化
        model.half().cuda()
    model = model.eval()

    text_list = [
         ( "图中的狗是什么品种？","../assets/demo.jpeg"),
         ( "这张图片的背景里有什么内容？","../assets/ghost.jpg"),
    ]
    for (input,image_path) in text_list:
        response, history = model.chat(tokenizer,image_path, input, history=[], max_length=2048,
                                       eos_token_id=config.eos_token_id,
                                       do_sample=True, top_p=0.7, temperature=0.95, )
        print("input", input)
        print("response", response)

    # response, history = base_model.chat(tokenizer, "图中的狗是什么品种？", history=[],max_length=30)
    # print('图中的狗是什么品种？',' ',response)

