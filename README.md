# large-model-trend-tracking

large-model-trend-tracking 大模型趋势追踪

## 文档介绍
```
这里整理近期发布的开源模型，模型主要分为三个领域：

一.自然语言处理领域；
二.计算机视觉领域；
三.多模态模型领域；
```

## 一.自然语言处理领域

1.ChatGLM
```
# (1)模型简介：
ChatGLM由清华大学唐杰团队开发。是一个开源的、支持中英双语的对话语言模型，基于 General Language Model(GLM)架构，具有62亿参数。
ChatGLM是一个具备问答和对话功能的语言模型，目前处于内测阶段，已经开启邀请制，并且将逐步扩大内测范围。
ChatGLM也已经开源了最新的中英双语对话模型ChatGLM-6B，结合模型量化技术，用户可以在消费级显卡上进行本地部署。
ChatGLM-6B共经过约1T标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术，模型参数达到了62亿。
虽然规模不及千亿模型，但是ChatGLM-6B已经能够生成相当符合人类偏好的回答，大大降低了用户部署的门槛。

ChatGLM-6B是一个开源的、支持中英双语问答的对话语言模型，并针对中文进行了优化。
该模型基于General Language Model（GLM）架构，具有62亿参数。
结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4量化级别下最低只需6GB显存）。

# (2)内测连接：
https://chatglm.cn/login

# (3)github:
https://github.com/THUDM/ChatGLM-6B

# (4)硬件支持：
最低6GB显存，推荐2080及以上的显卡；

# (5)相关介绍：
(1) 中文ChatGPT平替——ChatGLM：全新对话模型内测，手把手调教开源单卡版本_Chaos_Wang_的博客-CSDN博客. https://blog.csdn.net/qq_41667743/article/details/129539808 Accessed 3/22/2023.
(2) ChatGLM：千亿基座的对话模型启动内测，单卡版模型已全面开源 - 知乎. https://zhuanlan.zhihu.com/p/613862055 Accessed 3/22/2023.
(3) ChatGLM ：千亿基座的对话模型启动内测，单卡版模型已全面开源|大模型|chatglm_网易订阅. https://www.163.com/dy/article/HVQFBLBB0511FQO9.html Accessed 3/22/2023.
(4) GitHub - THUDM/ChatGLM-6B: ChatGLM-6B：开源双语对话语言模型 | An Open Bilingual .... https://github.com/THUDM/ChatGLM-6B Accessed 3/22/2023.
(5) 智谱AI开源单卡版模型ChatGLM-6B 已针对中文进行优化|序列|智谱ai|chatglm_网易订阅. https://www.163.com/dy/article/I0BRK9MU0511CP87.html Accessed 3/22/2023.
(6) 清华 ChatGLM-6B 中文对话模型部署简易教程_---Olive---的博客-CSDN博客. https://blog.csdn.net/qq_43475750/article/details/129665389 Accessed 3/22/2023.
```

- 2.LLaMA
```
# (1)模型简介：
LLaMA是Facebook Meta AI最新提出的语言模型，声称以更小的体积，在多数任务上超越了GPT-3的性能.
LLaMA是一个基于Transformer的语言模型，包括7B、13B、30B和65B四个版本，其中65B版本是目前最大的语言模型之一。
LLaMA的训练数据来自于公开的数据集，而非私有数据集，这使得LLaMA的训练过程更加透明和公正.
 
LLaMA的训练过程中，Facebook Meta AI使用了一些新的技术，
包括：
1）基于梯度的学习率调整策略；
2）基于梯度的模型剪枝策略；
3）基于梯度的模型量化策略；
4）基于梯度的模型蒸馏策略。
这些技术使得LLaMA的训练过程更加高效和稳定.
 
# (2)硬件支持：
资源占用：原版程序将消耗21G左右的显存，社区pyllama程序将消耗13G左右的显存。
建议配置：2张8G以上显存的显卡，推荐40系显卡；

# (3)github:
https://github.com/facebookresearch/llama

# (4)相关介绍：
(1) Meta最新模型LLaMA细节与代码详解_常鸿宇的博客-CSDN博客. https://blog.csdn.net/weixin_44826203/article/details/129255185 Accessed 3/22/2023.
(2) LLaMA: Open and Efficient Foundation Language Models. https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/ Accessed 3/22/2023.
(3) Introducing LLaMA: A foundational, 65-billion-parameter language model. https://ai.facebook.com/blog/large-language-model-llama-meta-ai/ Accessed 3/22/2023.
(4) GitHub - facebookresearch/llama: Inference code for LLaMA models. https://github.com/facebookresearch/llama Accessed 3/22/2023.
(5) LLaMA 开源语言模型7B 13B 30B 65B 泄漏版完整568GB国内网盘下载地址 免磁力链接 - openAI. https://openai.wiki/llama-model-download.html Accessed 3/22/2023.
(6) GitHub - shawwn/llama-dl: High-speed download of LLaMA, Facebook's 65B .... https://github.com/shawwn/llama-dl Accessed 3/22/2023.
(7) 模型杂谈：快速上手元宇宙大厂 Meta “开源泄露”的大模型（LLaMA）https://soulteary.com/2023/03/09/quick-start-llama-model-created-by-meta-research.html
(8) Facebook 强大的AI大型语言模型LLaMa遭泄露 https://www.chinaz.com/2023/0308/1503612.shtml
(9) 如何评价 LLaMA 模型泄露？https://www.zhihu.com/question/587479829/answer/2925378135
```

- 3.GPT-2
```
# (1)模型简介：
GPT-2是一种基于Transformer的大型语言模型，包含15亿个参数，是GPT模型的直接扩展，超过10倍的数据量上进行训练，参数量也多出了10倍。
GPT-2的训练数据集包含了800万个网页，共有40GB，这使得GPT-2可以处理不同领域不同任务的自然事件演示. 
GPT-2的训练目标是给定一个文本中前面的所有单词，预测下一个单词。GPT-2可以处理最长1024个单词的序列，
每个单词都会和它的前续路径一起「流过」所有的解码器模块. 

# (2)硬件支持：
建议配置：RTX20260以上，大于等于6GB显存；

# (2)github
https://github.com/openai/gpt-2

# (3)相关介绍：
(1) 迄今最大模型？OpenAI发布15亿参数量通用语言模型GPT-2 | 机器之心. https://www.jiqizhixin.com/articles/OpenAI-GPT-2 Accessed 3/22/2023.
(2) 逆天语言模型GPT-2最新开源：345M预训练模型和1.5B参数都来了 - 腾讯云开发者社区-腾讯云. https://cloud.tencent.com/developer/article/1425099 Accessed 3/22/2023.
(3) 完全图解GPT-2：看完这篇就够了（一） - 知乎. https://zhuanlan.zhihu.com/p/79714797 Accessed 3/22/2023.
(4) 上车！带你一文了解GPT-2模型（transformer语言模型可视化） - 知乎. https://zhuanlan.zhihu.com/p/95496557 Accessed 3/22/2023.
(5) GPT-4 - openai.com. https://openai.com/research/gpt-4 Accessed 3/22/2023.
```

- 4.复旦大学
```
# (1)模型简介：
MOSS是复旦大学自然语言处理实验室邱锡鹏教授团队发布的国内第一个对话式AI模型。
MOSS可执行对话生成、编程、事实问答等任务，打通了让生成式语言模型理解人类意图并具有对话能力的全部技术路径。
MOSS开发的基本步骤与ChatGPT一样，包括自然语言模型的基座训练、理解人类意图的对话能力训练两个阶段。

# (2)内测连接：
https://moss.fastnlp.top/
备注：为保证用户体验，MOSS 需要进行升级，现已停止服务；

# (3)github
https://github.com/TXSUN1997/MOSS
备注：截止目前2023-03-22没有公开相关代码及模型，披露未来会开源；

# (4)相关介绍：
(1) 复旦抢发类ChatGPT模型MOSS！被骂惨了 内测服务器挤崩. https://tech.ifeng.com/c/8NaOlCR6rW2 Accessed 3/22/2023.
(2) 复旦团队发布国内首个类ChatGPT模型MOSS，邀公众参与内测-36氪. https://www.36kr.com/p/2140789303986561 Accessed 3/22/2023.
(3) 复旦团队发布类ChatGPT模型——MOSS，邀公众参与内测被质疑 - 知乎. https://zhuanlan.zhihu.com/p/608793140 Accessed 3/22/2023.
(4) 复旦团队发布国内首个类 ChatGPT 模型 MOSS，将为国内大语言模型的探索和应用带来哪些影响? - 知乎. https://www.zhihu.com/question/585248111 Accessed 3/22/2023.
(5) 资讯｜复旦团队发布国内首个类ChatGPT模型MOSS，邀公众参与内测. https://fddi.fudan.edu.cn/5b/e2/c21257a482274/page.htm Accessed 3/22/2023.
(6) 复旦团队发布国内首个类ChatGPT模型MOSS，邀公众参与内测 - 知乎. https://zhuanlan.zhihu.com/p/608395901 Accessed 3/22/2023.
```

## 二.计算机视觉领域
- 1.Stable Diffusion
```
# (1)模型介绍：
Stable Diffusion是一种深度生成神经网络，是一种潜在扩散模型，由LMU Munich的CompVis小组和Runway开发。
它是一种文本到图像扩散模型，可以生成任何文本输入的照片般逼真的图像。Stable Diffusion是一个多功能模型，可以根据文本生成图像（text2img），
也可以根据图像生成文本（img2text）。它的优点是生成的图像质量更高、运行速度更快、消耗的资源以及内存占用更小。

# (2)官网:
https://stablediffusionweb.com/

# (3)硬件支持：
推荐：RTX 2060 显卡等6GB显存及以上配置，最低要求：至少4G显存，不能低于RTX1060；

# (3)github:
https://github.com/Stability-AI/stablediffusion
https://github.com/CompVis/stable-diffusion

# (4)相关介绍：
(1) 35张图，直观理解Stable Diffusion - 知乎. https://zhuanlan.zhihu.com/p/598999843 Accessed 3/22/2023.
(2) Stable Diffusion Online. https://stablediffusionweb.com/ Accessed 3/22/2023.
(3) Stable Diffusion - Wikipedia. https://en.wikipedia.org/wiki/Stable_Diffusion Accessed 3/22/2023.
(4) GitHub - Stability-AI/stablediffusion: High-Resolution Image Synthesis .... https://github.com/Stability-AI/StableDiffusion Accessed 3/22/2023.
(5) Stable Diffusion 2.0 Release — Stability AI. https://stability.ai/blog/stable-diffusion-v2-release Accessed 3/22/2023.
(6) 从零开始，手把手教你Window本地化部署stable diffusion AI绘图 https://zhuanlan.zhihu.com/p/578233719
```

## 三.多模态模型领域
- 1.UniDiffuser
```
# (1)模型简介：
清华大学朱军团队开源首个基于Transformer的多模态扩散大模型，支持文图互生、改写。
UniDiffuser是一个统一基于扩散模型模型的多模态概率框架，以原生扩散模型最小的改动，能够使用一个模型实现多个生成任务。
UniDiffuser可以进行半监督学习以及可以推广到更多的模态，这也是作者后续探索的方向. 
UniDiffuser能够显示地建模多模态数据中包括边缘分布、条件分布、联合分布在内的所有分布。
研究团队发现，关于不同分布的扩散模型学习都可以统一成一个视角：首先向两个模态的数据分别加入某种大小的噪声，然后再预测两个模态数据上的噪声.

# (2)paper
https://ml.cs.tsinghua.edu.cn/diffusion/unidiffuser.pdf

# (3)硬件支持：
最低10G显存，RTX 1080Ti及以上，推荐4090Ti；

# (4)github:
https://github.com/thu-ml/unidiffuser

# (5)相关介绍： 
(1) 清华朱军团队UniDiffuser论文阅读 - 知乎. https://zhuanlan.zhihu.com/p/613615521 Accessed 3/22/2023.
(2) 清华朱军团队开源UniDiffuser：首个基于Transformer的多模态扩散大模型！文图互生、改写全拿下！ - 知乎. https://zhuanlan.zhihu.com/p/614696522 Accessed 3/22/2023.
(3) GitHub - thu-ml/unidiffuser: Code and models for the paper "One .... https://github.com/thu-ml/unidiffuser Accessed 3/22/2023.
```

- 2.PaLM-E
```
# (1)模型简介：
PaLM-E是由谷歌和柏林工业大学共同开发的。PaLM-E是一种多模态具身视觉语言模型 (VLM)，能将视觉和语言集成到机器人控制中。
PaLM-E是一个解码器，只需给定前缀或提示，即可自回归地生成文本完成。
PaLM-E是一个通用的视觉语言模型，可以执行视觉任务，例如描述图像、检测对象或分类场景，也可以熟练地执行语言任务，
例如引用诗歌、解决数学方程或生成代码.

# (2)paper
https://palm-e.github.io/

# (3)备注信息：
项目代码及预训练模型截止2023-03-22没有开源；

# (4)相关介绍 
(1) PaLM-E: An Embodied Multimodal Language Model. https://palm-e.github.io/ Accessed 3/22/2023.
(2) PaLM-E: An embodied multimodal language model – Google AI Blog. https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html Accessed 3/22/2023.
(3) PaLM-E: An embodied multimodal language model – Google AI Blog. https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html Accessed 3/22/2023.
(4) 谷歌发布全球最大视觉语言模型 PaLM-E，5620 亿参数，几乎拥有所有语言能力，哪些信息值得关注？ - 知乎. https://www.zhihu.com/question/588441399 Accessed 3/22/2023.
```