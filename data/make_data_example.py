# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 12:50


import json


x1 = {
    "id": 0, "paragraph": [
        {
            "q": "<img>../assets/demo.jpeg</img>\n图中的狗是什么品种？",
            "a": "图中是一只拉布拉多犬。"
        }
    ]
}




x = [x1,]

with open('./finetune_train_examples.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for i in range(100):
        for j in range(len(x)):
            index += 1
            x[j]['id'] = index
            f.write(json.dumps(x[j],ensure_ascii=False) + '\n' )
