<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Daliy-Dialogue</h3>

  <p align="center">
   		由 Bloom 和 GPT 模型训练的日常对话上下文生成器
    <br />
  </p>
</p>

[In English](README_EN.md)

## 简要引述
[DailyDialog 日常对话](https://aclanthology.org/I17-1099/) 是一个高质量的多回合对话数据集。该项目提供了5个使用 Bloom 和 GPT 训练的日常对话上下文生成器。

## HuggingFace 展示

### 模型展示
|名称 |HuggingFace 模型链接| HuggingFace 空间链接 | 语言 |
|---------|--------|-------|-------|
|Bloom 英语日常对话生成器 🦅🌸| https://huggingface.co/svjack/bloom-daliy-dialogue-english | https://huggingface.co/spaces/svjack/bloom-daliy-dialogue-english | English |
|Bloom 中文日常对话生成器 🐰🌸| https://huggingface.co/svjack/bloom-daliy-dialogue | https://huggingface.co/spaces/svjack/bloom-daliy-dialogue-chinese | Chinese |
|GPT 中文日常对话生成器 🐰| https://huggingface.co/svjack/gpt-daliy-dialogue | https://huggingface.co/spaces/svjack/gpt-daliy-dialogue-chinese | Chinese |
|Bloom 中文对话生成器 🐰🌸| https://huggingface.co/svjack/bloom-dialogue | https://huggingface.co/spaces/svjack/bloom-dialogue-chinese | Chinese |
|GPT 中文对话生成器 🐰| https://huggingface.co/svjack/gpt-dialogue | https://huggingface.co/spaces/svjack/gpt-dialogue-chinese | Chinese |

### 由上述模型生成的数据集展示
|名称 |HuggingFace 数据集链接| HuggingFace 空间链接 | 语言 |
|---------|--------|-------|-------|
| 英语日常对话生成例子（提供搜索支持） 🦅🌸| https://huggingface.co/datasets/svjack/bloom-dialogue-generate-ds-en | https://huggingface.co/spaces/svjack/bloom-dialogue-english-sample-search| English |
| 中文对话生成例子（提供搜索支持） 🐰🌸| https://huggingface.co/datasets/svjack/bloom-dialogue-generate-ds-zh | https://huggingface.co/spaces/svjack/bloom-gpt-dialogue-chinese-sample-search | Chinese |

## 安装和结构

### 安装
```bash
pip install -r requirements.txt
```

### 结构

小技巧：减小max_length或温度参数，使得输出与你的输入问题更相关。

* 1 Bloom 英语每日问答生成器 🦅🌸:

```python
from predict import *
from transformers import BloomTokenizerFast, BloomForCausalLM

model_path = "svjack/bloom-daliy-dialogue-english"
tokenizer = BloomTokenizerFast.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)

obj = Obj(model, tokenizer)
obj.predict("This dog is fierce,", max_length=128)[0].split("\n-----\n")
```

将会输出:
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

* 2 Bloom 中文每日问答生成器 🐰🌸:

```python
from predict import *
from transformers import BloomTokenizerFast, BloomForCausalLM

model_path = "svjack/bloom-daliy-dialogue"
tokenizer = BloomTokenizerFast.from_pretrained(model_path)
model = BloomForCausalLM.from_pretrained(model_path)

obj = Obj(model, tokenizer)
obj.predict("这只狗很凶,", max_length=128)[0].split("\n-----\n")
```

将会输出:
```json
['这只狗很凶, 它是邻居的宠物。',
 '那是一只小狗。你知道，它不是一个很好的宠物。但正是因为它的行动。',
 '这倒是真的。我在它们还是小狗的时候就养了它们。我每个月只能负担300磅左右的费用。这就像一个大狗，就像电视上的专业厨师一样！',
 '绝对的 据说他们每天要供应3000只狗呢',
 '不可思议啊 对了，盘子里的这些东西是什么？',
 '哦，我的盘子，镂空的芝麻包，以及黄油和面粉。我将告诉你如何做一个搅拌']
```

<br/>

* 3 GPT 中文每日问答生成器 🐰:

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

将会输出:
```json
['这只狗很凶,你怎么知道的?',
 '我当然知道了，因为它是如此的凶。',
 '那是什么样的狗？',
 '他有一只小白猫哦，是吗?',
 '他还没有被抓到捕捕令呢我不相信他会抓到捕捕令他肯定在第二天早些时候就被抓到了好吧，我可以告诉他，但他必须在第二天早上给你打电话谢谢你对他的']
 ```

 你可以看到上面列表的最后一行太长了，没有很好地分段。<br/>
 通过使用 [svjack/GLM-Open-Dialogue](https://github.com/svjack/GLM-Open-Dialogue) 中的 [Context Reconstructor](https://huggingface.co/svjack/T5-dialogue-collect-v5)，我们可以尝试解决这个问题。


 ```python
 y = ['这只狗很凶,你怎么知道的?',
 '我当然知道了，因为它是如此的凶。',
 '那是什么样的狗？',
 '他有一只小白猫哦，是吗?',
 '他还没有被抓到捕捕令呢我不相信他会抓到捕捕令他肯定在第二天早些时候就被抓到了好吧，我可以告诉他，但他必须在第二天早上给你打电话谢谢你对他的']
 from reconstructor import *
 predict_split(y)
 ```

 将会输出:
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

## 更多信息和讨论
你可以看到 Bloom 在上下文分割方面表现得更好。<br/><br/>
可以从 https://github.com/svjack/GLM-Open-Dialogue 获取更多信息，那里包括获取上下文重构器以及有关开放式对话上下文生成器的主题。

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
