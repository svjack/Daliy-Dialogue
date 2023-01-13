from predict import *
from transformers import BloomTokenizerFast, BloomForCausalLM

import os
import gradio as gr

model_path = "svjack/bloom-daliy-dialogue-english"
tokenizer = BloomTokenizerFast.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)

obj = Obj(model, tokenizer)

example_sample = [
    ["This dog is fierce,", 128],
    ["Do you like this film?", 64],
]

def demo_func(prefix, max_length):
    max_length = max(int(max_length), 32)
    l = obj.predict(prefix, max_length=max_length)[0].split("\n-----\n")
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
        title=f"Bloom English Daliy Dialogue Generator 🦅🌸 demonstration",
        examples=example_sample if example_sample else None,
        cache_examples = False
    )

demo.launch(server_name=None, server_port=None)
