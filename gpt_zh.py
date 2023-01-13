from predict import *
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

model_path = "svjack/gpt-daliy-dialogue"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

obj = Obj(model, tokenizer)
x = obj.predict("这只狗很凶,", max_length=128)[0]
list(map(lambda x: "".join(x).replace(" ", ""),batch_as_list(re.split(r"([。.？?])" ,x), 2)))
'''
['这只狗很凶,你怎么知道的?',
 '我当然知道了，因为它是如此的凶。',
 '那是什么样的狗？',
 '他有一只小白猫哦，是吗?',
 '他还没有被抓到捕捕令呢我不相信他会抓到捕捕令他肯定在第二天早些时候就被抓到了好吧，我可以告诉他，但他必须在第二天早上给你打电话谢谢你对他的']
'''

x = obj.predict("你饿吗？", max_length=128)[0]
list(map(lambda x: "".join(x).replace(" ", ""),batch_as_list(re.split(r"([。.？?])" ,x), 2)))
'''
['你饿吗？',
 '是的，我想吃点东西。',
 '你想去什么地方吃午饭吗？',
 '我也不知道啊去一家意大利餐厅怎么样?',
 '哦，我很喜欢意大利菜。',
 '那里的食物怎么样?',
 '非常好。',
 '我还不知道呢他们提供的是意大利菜。',
 '我以前听说他们在街边的餐厅里有一家新的意大利餐厅。',
 '我也想去那里']
'''

from reconstructor import *

obj = Obj(model, tokenizer)
x = obj.predict("这只狗很凶,", max_length=128)[0]
y = list(map(lambda x: "".join(x).replace(" ", ""),batch_as_list(re.split(r"([。.？?])" ,x), 2)))
y
'''
['这只狗很凶,你怎么知道的?',
 '我当然知道了，因为它是如此的凶。',
 '那是什么样的狗？',
 '他有一只小白猫哦，是吗?',
 '他还没有被抓到捕捕令呢我不相信他会抓到捕捕令他肯定在第二天早些时候就被抓到了好吧，我可以告诉他，但他必须在第二天
'''
predict_split(y)
'''
['这只狗很凶,你怎么知道的?',
 '我当然知道了,因为它是如此的凶。',
 '那是什么样的狗?',
 '他有一只小白猫',
 '哦,是吗?',
 '他还没有被捕捉到捕捕令呢',
 '我不相信他会抓到它',
 '捕捕令?他肯定在第二天早些时候就被抓到了',
 '好吧,我可以告诉他,但他必须在第二天早上给你打电话。',
 '谢谢你对他的到来']
'''
