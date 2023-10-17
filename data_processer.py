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
    sliding = 2




def build_inputs_with_image(self, tokenizer, image_path: str, query: str, history: List[Tuple[str, str]] = None):
    image_path = image_path.strip()
    if image_path:
        prompt = "<img>{}</img>".format(image_path)
    else:
        prompt = ""
    for i, (old_query, response) in enumerate(history):  # history removes image urls/paths, while query does not.
        prompt += "问：{}\n答：{}\n".format(old_query, response)
    prompt += "问：{}\n答：".format(query)
    prompt, image_position, torch_image = self.process_image(prompt)
    if torch_image is not None:
        torch_image = torch_image.to(self.dtype).to(self.device)
        input0 = tokenizer.encode(prompt[:image_position], add_special_tokens=False)
        input1 = [tokenizer.unk_token_id] * self.image_length
        input2 = tokenizer.encode(prompt[image_position:], add_special_tokens=False)
        inputs = sum([input0, input1, input2], [])
        inputs = {
            "input_ids": torch.tensor([tokenizer.build_inputs_with_special_tokens(inputs)], dtype=torch.long).to(
                self.device),
            "pre_image_length": len(input0),
            "images": torch_image}
    else:
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        inputs["pre_image_length"] = 0
    return inputs



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

from deep_training.nlp.models.visualglm.visual import BlipImageEvalProcessor
from PIL import Image
from io import BytesIO
class TokenIdsMaker:

    @classmethod
    def process_image(cls, text, image=None):
        '''Process image in text.
        Args:
            text: str, text.
            image: Optional, image path / url / PIL image.
        '''
        image_position = text.rfind("<img>") + 5
        # extract path from <img></img> using re
        image_path = re.findall(r"<img>(.*?)</img>", text)
        image_path = image_path[-1] if image_path else None
        if image_path is not None:
            assert image is None, "image and image_path cannot be both not None."
            text = text.replace(f"<img>{image_path}</img>", "<img></img>")
            # url
            # if image_path.startswith("http"):
            #     response = requests.get(image_path, timeout=10)
            #     image = Image.open(BytesIO(response.content))
            # # local path
            # else:
            #     image = Image.open(image_path)
            image = Image.open(image_path)
        if image is not None:
            processor = BlipImageEvalProcessor(224)
            image = processor(image.convert('RGB'))
            image = image.unsqueeze(0)
        return text, image_position, image
    @classmethod
    def final(cls, input_ids: typing.List, sptoken, max_seq_length, tokenizer):
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
            'ctxlen': ctxlen
        }
        return d
    @classmethod
    def tunction(cls, tokenizer: ChatGLMTokenizer,config, examples, max_seq_length, sptoken: typing.List):

        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            prompt_text = build_template(q, prefix=prefix, history=examples[:sid])
            text, image_position, image = cls.process_image(text=prompt_text)

            torch_image = torch_image.to(self.dtype).to(self.device)
            input0 = tokenizer.encode(prompt[:image_position], add_special_tokens=False)
            input1 = [tokenizer.unk_token_id] * self.image_length
            input2 = tokenizer.encode(prompt[image_position:], add_special_tokens=False)
            inputs = sum([input0, input1, input2], [])
            inputs = {
                "input_ids": torch.tensor([tokenizer.build_inputs_with_special_tokens(inputs)], dtype=torch.long).to(
                    self.device),
                "pre_image_length": len(input0),
                "images": torch_image}


            a_ids = tokenizer.encode(text=,add_special_tokens=False)
            b_ids = tokenizer.encode(text=a, add_special_tokens=False)
            while len(a_ids) + len(b_ids) > max_seq_length - len(sptoken) - 1:
                if len(b_ids) > len(a_ids):
                    b_ids.pop(-1)
                else:
                    a_ids.pop(0)
            b_ids += [config.eos_token_id]
            input_ids = a_ids + sptoken + b_ids
            assert len(input_ids) <= max_seq_length
            ds.append(cls.final(input_ids, sptoken, max_seq_length, tokenizer))
        return ds


    @classmethod
    def slidding(cls, tokenizer: ChatGLMTokenizer,config, examples, max_seq_length, sptoken: typing.List,
                 sliding_size=None,
                 src_max_length=-1,
                 dst_max_length=-1,p=1):

        if sliding_size is None or sliding_size < 0:
            sliding_size = max_seq_length - len(sptoken)

        assert sliding_size <= max_seq_length - len(sptoken)

        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids = tokenizer.encode(text=build_template(q, prefix=prefix,history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a, add_special_tokens=False) + [config.eos_token_id]

            if src_max_length and src_max_length > 0:
                a_ids = a_ids[:src_max_length]
            if dst_max_length and dst_max_length > 0:
                b_ids = b_ids[:dst_max_length]

            b_ids += [config.eos_token_id]

            input_ids_qa = a_ids + sptoken + b_ids
            a_length = len(a_ids)
            pos = 0
            while pos < len(input_ids_qa):
                if pos + max_seq_length <= a_length:
                    input_ids = input_ids_qa[pos:pos + max_seq_length - 2]
                    if p > 0:
                        input_ids = input_ids[0:-p] + sptoken + input_ids[-p:]
                    else:
                        p = random.randint(0, max_seq_length - 2)
                        input_ids = input_ids[0:p] + sptoken + input_ids[p:]

                elif sptoken[0] in input_ids_qa[pos:pos + max_seq_length]:
                    val = input_ids_qa[pos:pos + max_seq_length][-1]
                    if val == sptoken[-1]:
                        input_ids = input_ids_qa[pos + 1:pos + max_seq_length + 1]

                    elif val == sptoken[0]:
                        input_ids = input_ids_qa[pos + 2:pos + max_seq_length + 2]
                    else:
                        input_ids = input_ids_qa[pos:pos + max_seq_length]
                else:
                    input_ids = sptoken + input_ids_qa[pos:pos + max_seq_length - 2]
                pos += sliding_size
                ds.append(cls.final(input_ids, sptoken, max_seq_length, tokenizer))
            return ds
