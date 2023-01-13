from predict import *
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
)
import jieba.posseg as posseg

model_path = "svjack/T5-dialogue-collect-v5"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

rec_obj = Obj(model, tokenizer)

def process_one_sent(input_):
    assert type(input_) == type("")
    input_ = " ".join(map(lambda y: y.word.strip() ,filter(lambda x: x.flag != "x" ,
    posseg.lcut(input_))))
    return input_

def predict_split(sp_list, cut_tokens = True):
    assert type(sp_list) == type([])
    if cut_tokens:
        src_text = '''
            根据下面的上下文进行分段：
            上下文：{}
            答案：
            '''.format(" ".join(
            map(process_one_sent ,sp_list)
            ))
    else:
        src_text = '''
            根据下面的上下文进行分段：
            上下文：{}
            答案：
            '''.format("".join(sp_list))
    print(src_text)
    pred = rec_obj.predict(src_text)[0]
    pred = list(filter(lambda y: y ,map(lambda x: x.strip() ,pred.split("分段:"))))
    return pred
