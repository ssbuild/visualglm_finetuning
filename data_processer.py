# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import random
import re
import typing
from enum import Enum
import numpy as np
import requests

from aigc_zoo.model_zoo.visualglm.llm_model import ChatGLMTokenizer


class DataStrategy(Enum):
    truncation = 1






def build_template_chatglm(query, answer = None,prefix=None, image_path = None, history=None):
    if image_path:
        prompt = "<img>{}</img>".format(image_path)
    else:
        prompt = ""
    prompt += prefix or ''
    sid = 0
    if history is not None:
        for q, a in history:
            prompt += "[Round {}]\n问：{}\n答：{}".format(sid,q, a)
            sid += 1
    prompt += query if sid == 0 else "[Round {}]\n问：{}\n答：".format(sid, query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_chatglm2(query, answer = None,prefix=None, image_path = None,history=None):
    if image_path:
        prompt = "<img>{}</img>".format(image_path)
    else:
        prompt = ""
    prompt += prefix or ''
    sid = 1
    if history is not None:
        for q, a in history:
            prompt += "[Round {}]\n问：{}\n答：{}".format(sid,q, a)
            sid += 1
    prompt += "[Round {}]\n问：{}\n答：".format(sid, query)
    if answer is not None:
        prompt += answer
    return prompt


def build_template_default(query, answer = None,prefix=None, image_path = None, history=None):
    if image_path:
        prompt = "<img>{}</img>".format(image_path)
    else:
        prompt = ""
    prompt += prefix or ''
    if history is not None:
        for q,a in history:
            prompt += "User: {}\nAssistant:{}".format(q,a)
    prompt += "User: {}\nAssistant:".format(query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_tiger(query,answer = None,prefix=None, image_path = None, history=None):
    if image_path:
        prompt = "<img>{}</img>".format(image_path)
    else:
        prompt = ""
    prompt += prefix or ''
    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    if history is not None:
        for q,a in history:
            prompt += "{}{}{}{}".format(tok_ins,q,tok_res,a)

    prompt += "{}{}{}".format(tok_ins, query, tok_res)
    if answer is not None:
        prompt += answer
    return prompt


# 切换模版
build_template = build_template_chatglm


class TokenIdsMaker:

    @classmethod
    def process_image(cls, text):
        '''Process image in text.
        Args:
            text: str, text.
            image: Optional, image path / url / PIL image.
        '''
        image_position = text.rfind("<img>") + 5
        # extract path from <img></img> using re
        image_path = re.findall(r"<img>(.*?)</img>", text)
        image_path = image_path[-1] if image_path else None
        text = text.replace(f"<img>{image_path}</img>", "<img></img>")
        return text, image_position,image_path
    @classmethod
    def final(cls, input_ids: typing.List, sptoken, max_seq_length, tokenizer,pre_image_length,image_path):
        ctxlen = input_ids.index(sptoken[-1])
        mask_position = ctxlen - 1
        labels = [-100] * ctxlen + input_ids[mask_position + 1:]

        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        ctxlen = np.asarray(ctxlen, dtype=np.int32)
        if pad_len:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))


        d = {
            'input_ids': input_ids,
            'labels': labels,
            'seqlen': seqlen,
            'ctxlen': ctxlen,
            'pre_image_length': np.asarray(pre_image_length,dtype=np.int32),
            'image_path': np.asarray(bytes(image_path,encoding='utf-8'))
        }
        return d
    @classmethod
    def tunction(cls, tokenizer: ChatGLMTokenizer,config, examples, max_seq_length, sptoken: typing.List):
        image_length = config.image_length
        assert image_length < max_seq_length
        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            prompt = build_template(q, prefix=prefix, history=examples[:sid])
            prompt, image_position,image_path = cls.process_image(text=prompt)

            assert image_path
            input0 = tokenizer.encode(prompt[:image_position], add_special_tokens=False)
            input1 = [tokenizer.unk_token_id] * image_length
            input2 = tokenizer.encode(prompt[image_position:], add_special_tokens=False)
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=a, add_special_tokens=False)
            pre_image_length = len(input0)
            while len(a_ids) + len(b_ids) > max_seq_length - len(sptoken) - 1:
                if len(a_ids) <= pre_image_length + image_length:
                    b_ids.pop(-1)
                else:
                    if len(b_ids) > len(a_ids):
                        b_ids.pop(-1)
                    else:
                        a_ids.pop(0)
            b_ids += [config.eos_token_id]
            input_ids = a_ids + sptoken + b_ids
            assert len(input_ids) <= max_seq_length
            ds.append(cls.final(input_ids, sptoken, max_seq_length, tokenizer,pre_image_length,image_path))
        return ds

