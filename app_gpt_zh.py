from predict import *
from reconstructor import *
from transformers import BertTokenizer, GPT2LMHeadModel

import os
import gradio as gr

model_path = "svjack/gpt-daliy-dialogue"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

obj = Obj(model, tokenizer)

example_sample = [
    ["这只狗很凶,", 128],
    ["你饿吗？", 128],
]

def demo_func(prefix, max_length, use_pred_sp = True):
    max_length = max(int(max_length), 32)
    x = obj.predict(prefix, max_length=max_length)[0]
    y = list(map(lambda x: "".join(x).replace(" ", ""),batch_as_list(re.split(r"([。.？?])" ,x), 2)))
    if use_pred_sp:
        l = predict_split(y)
    else:
        l = y
    l_ = []
    for ele in l:
        if ele not in l_:
            l_.append(ele)
    l = l_
    assert type(l) == type([])
    return {
        "Dialogue Context": l
    }

demo = gr.Interface(
        fn=demo_func,
        inputs=[gr.Text(label = "Prefix"),
                gr.Number(label = "Max Length", value = 128)
        ],
        outputs="json",
        title=f"GPT Chinese Daliy Dialogue Generator 🐰 demonstration",
        examples=example_sample if example_sample else None,
        cache_examples = False
    )

demo.launch(server_name=None, server_port=None)
