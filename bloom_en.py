from predict import *
from transformers import BloomTokenizerFast, BloomForCausalLM

model_path = "svjack/bloom-daliy-dialogue-english"
tokenizer = BloomTokenizerFast.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)

obj = Obj(model, tokenizer)
obj.predict("This dog is fierce,", max_length=128)[0].split("\n-----\n")
'''
[
    "This dog is fierce, nowadays. ",
    " What's his name? ",
    " His name is Bingo. ",
    " What kind of dog is he? ",
    " We're not sure because the neighbour gave him to us after they moved away from here. ",
    " Well, he sure likes to chew my father's shoes when he likes to scratch the couch. ",
    " Is he well behaved? ",
    " Yeah, he likes to scratch the couch but he likes to scratch the couch regularly. ",
    " He likes to scratch the couch, doesn't he? ",
    " Yes, he likes to scratch the couch"
  ]
'''

obj.predict("Do you like this film?", max_length=64)[0].split("\n-----\n")
'''
[
    "Do you like this film?it? ",
    " I like it, but it's not as crowded as we excepted today. ",
    " It does have a lot of girls in it. ",
    " True. True, too. ",
    " Do you think Bruce Willis just drank Coke for his health"
]
'''
