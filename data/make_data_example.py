# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 12:50
import json



some = [
    {"img": "../assets/demo.jpeg", "prompt": "图中的狗是什么品种？", "label": "图中是一只拉布拉多犬。"},
    {"img": "../assets/2p.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是蒙蒙细雨。"},
    {"img": "../assets/pig.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是是虚化的。"},
    {"img": "../assets/meme.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是蓝色的木质地板。"},
    {"img": "../assets/passport.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是棕黄色木质桌子。"},
    {"img": "../assets/tower.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是黄昏的天空、云彩和繁华的城市高楼。"},
    {"img": "../assets/rub.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是太阳、大树、蓝天白云。"},
    {"img": "../assets/push.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是蓝天和沙漠。"},
    {"img": "../assets/traf.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是城市街道。"},
    {"img": "../assets/music.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是一个音乐混音器。"},
    {"img": "../assets/pattern.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是小区的楼房和街道。"},
    {"img": "../assets/rou.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是大理石桌子和一个盘子。"},
    {"img": "../assets/katong.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是绿色的草地。"},
    {"img": "../assets/man.jpg", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是城市的街道和高楼。"},
    {"img": "../assets/kobe.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是虚化的观众席。"},
    {"img": "../assets/panda.jpg", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是纯白的。"},
    {"img": "../assets/titan.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是一座雕像。"},
    {"img": "../assets/woman.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是纯蓝的。"},
    {"img": "../assets/ghost.jpg", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是一个房间。"},
    {"img": "../assets/justice.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是天空和阳光。"},
    {"img": "../assets/tianye.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是金黄的田野。"}
]

x = []
for i,item in enumerate(some):
    jd = {
        "id": i,
        "paragraph": [
            {
                "q": '<img>' + item["img"] + '</img>\n' +  item["prompt"],
                "a": item["label"],
            }
        ]
    }
    x.append(jd)
    



with open('./finetune_train_examples.json',mode='w',encoding='utf-8',newline='\n') as f:
    index = 0
    for i in range(20):
        for j in range(len(x)):
            index += 1
            x[j]['id'] = index
            f.write(json.dumps(x[j],ensure_ascii=False) + '\n' )
