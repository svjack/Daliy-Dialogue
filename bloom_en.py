from predict import *
from transformers import BloomTokenizerFast, BloomForCausalLM

model_path = "svjack/bloom-daliy-dialogue"
tokenizer = BloomTokenizerFast.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)

obj = Obj(model, tokenizer)
obj.predict("This dog is fierce,", max_length=128)[0].split("\n-----\n")
'''
['这只狗很凶, 它是邻居的宠物。',
 '那是一只小狗。你知道，它不是一个很好的宠物。但正是因为它的行动。',
 '这倒是真的。我在它们还是小狗的时候就养了它们。我每个月只能负担300磅左右的费用。这就像一个大狗，就像电视上的专业厨师一样！',
 '绝对的 据说他们每天要供应3000只狗呢',
 '不可思议啊 对了，盘子里的这些东西是什么？',
 '哦，我的盘子，镂空的芝麻包，以及黄油和面粉。我将告诉你如何做一个搅拌']
'''

obj.predict("Are you hungry?", max_length=128)[0].split("\n-----\n")
'''
['你饿吗？',
 '没有，但我想尝尝中国菜。',
 '我不知道你喜欢做什么',
 '我很喜欢吃中国菜，但我更喜欢西餐。',
 '好吧，说实话，我不太喜欢。你知道，我总是做这个工作，我希望能有机会去一家新的中国餐厅吃饭。',
 '那么，你想去哪个餐厅？',
 '我也不知道啊 第一次来这里',
 '让我看看... 这是个不错的餐厅',
 '是的，它很明亮，而且它是一个新的装修。它可能适合所有年龄段段的东西',
 '这听起来不错。你要多少钱？',
 '360']
'''
