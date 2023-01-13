<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Daliy-Dialogue</h3>

  <p align="center">
   		A Daliy Dialogue Context Generator trained with Bloom and GPT
    <br />
  </p>
</p>

## Brief introduction
[DailyDialog](https://aclanthology.org/I17-1099/) is a high-quality multi-turn dialog dataset. This project give three Daliy Dialogue Context Generators trained with Bloom and GPT.

## HuggingFace Space demonstration
* 1 Bloom English Daliy Dialogue Generator 🦅🌸 demonstration: https://huggingface.co/spaces/svjack/bloom-daliy-dialogue-english
* 2 Bloom Chinese Daliy Dialogue Generator 🐰🌸 demonstration: https://huggingface.co/spaces/svjack/bloom-daliy-dialogue-chinese
* 3 GPT Chinese Daliy Dialogue Generator 🐰 demonstration: https://huggingface.co/spaces/svjack/gpt-daliy-dialogue-chinese

## Installation and Instructions

### Installation
```bash
pip install -r requirements.txt
```
### Instructions

Tips: try to decrease the max_length, the output may more related with the text you ask.

* 1 Bloom English Daliy Dialogue Generator 🦅🌸:

```python
from predict import *
from transformers import BloomTokenizerFast, BloomForCausalLM

model_path = "svjack/bloom-daliy-dialogue-english"
tokenizer = BloomTokenizerFast.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)

obj = Obj(model, tokenizer)
obj.predict("This dog is fierce,", max_length=128)[0].split("\n-----\n")
```

will output:
```json
['This dog is fierce, nowadays. ',
  " What's his name? ",
  ' His name is Bingo. ',
  ' What kind of dog is he? ',
  " We're not sure because the neighbour gave him to us after they moved away from here. ",
  " Well, he sure likes to chew my father's shoes when he likes to scratch the couch. ",
  ' Is he well behaved? ',
  ' Yeah, he likes to scratch the couch but he likes to scratch the couch regularly. ',
  " He likes to scratch the couch, doesn't he? ",
  ' Yes, he likes to scratch the couch']
```

<br/>

* 2 Bloom Chinese Daliy Dialogue Generator 🐰🌸:

```python
from predict import *
from transformers import BloomTokenizerFast, BloomForCausalLM

model_path = "svjack/bloom-daliy-dialogue"
tokenizer = BloomTokenizerFast.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)

obj = Obj(model, tokenizer)
obj.predict("这只狗很凶,", max_length=128)[0].split("\n-----\n")
```

will output:
```json
['这只狗很凶, 它是邻居的宠物。',
 '那是一只小狗。你知道，它不是一个很好的宠物。但正是因为它的行动。',
 '这倒是真的。我在它们还是小狗的时候就养了它们。我每个月只能负担300磅左右的费用。这就像一个大狗，就像电视上的专业厨师一样！',
 '绝对的 据说他们每天要供应3000只狗呢',
 '不可思议啊 对了，盘子里的这些东西是什么？',
 '哦，我的盘子，镂空的芝麻包，以及黄油和面粉。我将告诉你如何做一个搅拌']
```

<br/>

* 3 GPT Chinese Daliy Dialogue Generator 🐰:

```python
from predict import *
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

model_path = "svjack/gpt-daliy-dialogue"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

obj = Obj(model, tokenizer)
x = obj.predict("这只狗很凶,", max_length=128)[0]
list(map(lambda x: "".join(x).replace(" ", ""),batch_as_list(re.split(r"([。.？?])" ,x), 2)))
```

will output:
```json
['这只狗很凶,你怎么知道的?',
 '我当然知道了，因为它是如此的凶。',
 '那是什么样的狗？',
 '他有一只小白猫哦，是吗?',
 '他还没有被抓到捕捕令呢我不相信他会抓到捕捕令他肯定在第二天早些时候就被抓到了好吧，我可以告诉他，但他必须在第二天早上给你打电话谢谢你对他的']
 ```

 You can see the last line of above list is too long, that not well segmented.<br/>
 And with the help of [Context Reconstructor](https://huggingface.co/svjack/T5-dialogue-collect-v5) in [svjack/GLM-Open-Dialogue](https://github.com/svjack/GLM-Open-Dialogue), we can try to fix this problem.
 ```python
 y = ['这只狗很凶,你怎么知道的?',
 '我当然知道了，因为它是如此的凶。',
 '那是什么样的狗？',
 '他有一只小白猫哦，是吗?',
 '他还没有被抓到捕捕令呢我不相信他会抓到捕捕令他肯定在第二天早些时候就被抓到了好吧，我可以告诉他，但他必须在第二天早上给你打电话谢谢你对他的']
 from reconstructor import *
 predict_split(y)
 ```

 will output:
 ```json
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
 ```

## More Info and Disscussion
You can see Bloom may perform better in context segmentation.<br/><br/>
More information can be find from https://github.com/svjack/GLM-Open-Dialogue to get [Context Reconstructor](https://huggingface.co/svjack/T5-dialogue-collect-v5) and topic about Open Dialogue Context Generator.

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - https://huggingface.co/svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/Daliy-Dialogue](https://github.com/svjack/Daliy-Dialogue)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
-->
* [Bigscience](https://bigscience.huggingface.co)
* [TextBox](https://github.com/RUCAIBox/TextBox)
* [Langboat](https://huggingface.co/Langboat)
* [uer](https://huggingface.co/uer)
* [ClueAI](https://huggingface.co/ClueAI)
* [DialoGPT](https://github.com/microsoft/DialoGPT)
* [svjack/GLM-Open-Dialogue](https://github.com/svjack/GLM-Open-Dialogue)
* [svjack](https://huggingface.co/svjack)
