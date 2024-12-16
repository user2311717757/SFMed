# SFMed

SFMed is a large-scale medical domain model that has undergone continued pre-training, supervised fine-tuning, and alignment based on Qwen2-7B-Instruct. Its current performance has comprehensively surpassed that of other open-source medical models and is comparable to proprietary models.


The core functions of SFMed include:

- Medical Consultation: It can act as a doctor to answer users' questions about diseases and other health-related issues. This includes single-turn question-and-answer dialogues as well as multi-turn dialogues with follow-up questions.

- Pharmaceutical Consultation: It understands medical terminology, drug names, and other specialized terms, providing precise professional knowledge in the medical field.

## Update Log

[2024/11/01] 🚀[Open-source SFMed and 7B model weights🤗](https://huggingface.co/) #todo (Once  the  paper  is  accepted,  we  will  release  our  model.)

### Download

[SFMed-7B-Instruct](https://huggingface.co/) #todo

### Inference

Same inference method as [🤗Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)

## Training Data

A total of approximately2.7B tokens, including around1.4B tokens of general corpus and about1.3B tokens from the medical domain.

- 中文医疗数据

| 数据集名称                    | 数据集简介                                                                                                                                                               | 数据集条数 | 数据集Token数 |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ------------- |
| CMtMedQA                      | 包含 70,000 条多轮对话数据集，来源于真实医患交流，包含了大量的主动问询语句                                                                                               | 68023      | 25569575      |
| ChatMed_Consult_Dataset       | 来自于互联网上的医疗问诊问题，反映了真实世界的不同用户/患者的医疗问诊需求。目前response都是由OpenAI GPT-3.5引擎回答的                                                    | 549326     | 79788168      |
| DISC-Med-SFT                  | 包含了超过47万个衍生于现有的医疗数据集重新构建得到的样本。                                                                                                               | 464898     | 147775556     |
| MedDiag                       | 医患之间的中文对话                                                                                                                                                       | 2725990    | 351973718     |
| PromptCBLUE                   | 对CBLUE基准进行二次开发，将16种不同的医疗场景NLP任务全部转化为基于提示的语言生成任务,形成首个中文医疗场景的LLM评测基准                                                   | 151500     | 27127160      |
| ShenNong_TCM                  | 以中医药知识图谱为基础,采用以实体为中心的自指令方法，调用ChatGPT得到11w+的围绕中医药的指令数据；                                                                         | 112565     | 25168926      |
| cMedQA-V2.0                   | 中国社区医疗问答的数据集                                                                                                                                                 | 226266     | 22608102      |
| huatuo_knowledge_graph_qa     | 基于医学知识图谱构建了这个QA数据集，共有798444条数据，其中问题是通过模板构建的，答案是知识图谱中条目的内容。                                                             | 798444     | 30869480      |
| huatuo_sft_train_data         | 训练HuatuoGPT的部分数据                                                                                                                                                  | 226042     | 67348069      |
| huatuo_encyclopedia_qa        | 从医学百科全书和医学文章中提取医学QA对。在中文维基百科上收集了8699个疾病百科全书条目和2736个药物百科全书条目。此外，我们从千问健康网站上抓取了226432篇高质量的医学文章。 | 362420     | 135298899     |
| webMedQA                      | 从在线健康咨询网站收集的真实的中文医疗问答的数据集                                                                                                                       | 316110     | 56211345      |
| Chinese-medical-dialogue-data | 中文医疗问答数据集                                                                                                                                                       | 792099     | 115022641     |
| CMExam                        | 来自中国国家医师资格考试的数据集。包括60K以上的多项选择题                                                                                                                | 68119      | 15019251      |
| CMB-Exam                      | 全方位多层次测评模型医疗知识                                                                                                                                             | 280839     | 18724518      |
| questions                     | 一些中文医疗数据                                                                                                                                                         | 48376      | 4376377       |
| 教科书                        | 中文医学教科书数据                                                                                                                                                       | 21471      | 15179698      |
| Crawler                       | 从好大夫、丁香园等自行爬取的数据                                                                                                                                         | 482922     | 68760072      |
| yiigle                        | 中华医学期刊标题与摘要                                                                                                                                                   | 308869     | 104137233     |

- 中文通用数据

| 数据集名称    | 数据集简介                                                                           | 数据集条数 | 数据集Token数 |
| ------------- | ------------------------------------------------------------------------------------ | ---------- | ------------- |
| baike2018qa   | 百科类问答，含有150万个预先过滤过的、高质量问题和答案                                | 1470142    | 358976186     |
| webtext2019zh | 社区问答，含有410万个预先过滤过的、高质量问题和回复。                                | 4258310    | 763721567     |
| wiki2019zh    | 维基百科，可以做为通用中文语料，做预训练的语料或构建词向量，也可以用于构建知识问答。 | 1248027    | 314398588     |

## 评测

评测数据集：

评测数据集包括多轮对话、单轮对话、医学名词解释及选择题数据

- CMtMedQA：[https://github.com/SupritYoung/Zhongjing/blob/main/data/CMtMedQA_test.json](https://github.com/SupritYoung/Zhongjing/blob/main/data/CMtMedQA_test.json)
- huatuo26M：[https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets](https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets)
- webMedQA：[https://github.com/hejunqing/webMedQA](https://github.com/hejunqing/webMedQA)
- medtiku：[https://www.medtiku.com/](https://www.medtiku.com/)
- 选择题：从CEval等抽取得到的与医疗领域相关的选择题数据

评测方式：

下载评测数据集，替换模型路径后，运行下面的代码即可。

依赖库：[SWIFT](https://github.com/modelscope/ms-swift)

```python
python infer_vllm.py --model_path 模型路径
```

### 1. 与权威医疗开源模型的对比结果

| QA-Rouge     |           | SFMed vs. HuatuoGPT-II      | SFMed vs. Zhongjing            | SFMed vs. ChiMed-GPT           | SFMed vs. WiNGPT2           |
| ------------ | --------- | --------------------------- | ------------------------------ | ------------------------------ | --------------------------- |
| 多轮对话     | CMtMedQA  | **0.7544/0.0/0.2456** | **0.5919/0.0019/0.4062** | **0.8607/0.0019/0.1373** | 0.3675/0.0/0.6325           |
| 单轮对话     | All       | **0.558/0.008/0.434** | **0.539/0.014/0.447**    | **0.5/0.013/0.487**      | **0.545/0.01/0.445**  |
|              | huatuo26M | **0.584/0.008/0.408** | **0.506/0.016/0.478**    | **0.506/0.008/0.486**    | **0.532/0.008/0.46**  |
|              | webMedQA  | **0.532/0.008/0.46**  | **0.572/0.012/0.416**    | **0.494/0.018/0.488**    | **0.558/0.012/0.43**  |
| 医学名词解释 | medtiku   | **0.76/0.003/0.237**  | **0.638/0.002/0.36**     | **0.687/0.005/0.308**    | **0.654/0.001/0.345** |

| QA-GPT   |           | SFMed vs. HuatuoGPT-II         | SFMed vs. Zhongjing            | SFMed vs. ChiMed-GPT           | SFMed vs. WiNGPT2              |
| -------- | --------- | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------ |
| 多轮对话 | CMtMedQA  | **0.3907/0.4874/0.1219** | **0.6209/0.3366/0.0426** | **0.9884/0.0097/0.0019** | **0.6209/0.3133/0.0658** |
| 单轮对话 | All       | **0.242/0.567/0.186**    | **0.702/0.248/0.045**    | **0.975/0.013/0.005**    | **0.861/0.114/0.019**    |
|          | huatuo26M | **0.282/0.54/0.178**     | **0.712/0.244/0.044**    | **0.972/0.016/0.008**    | **0.876/0.106/0.014**    |
|          | webMedQA  | **0.202/0.594/0.194**    | **0.692/0.252/0.046**    | **0.978/0.01/0.002**     | **0.846/0.122/0.024**    |

| 选择题 | HuatuoGPT-II | Zhongjing | ChiMed-GPT | WiNGPT2 | SFMed            |
| ------ | ------------ | --------- | ---------- | ------- | ---------------- |
| All    | 0.1844       | 0.1195    | 0.3348     | 0.3075  | **0.7596** |
| PLE    | 0.11         | 0.085     | 0.21       | 0.265   | **0.685**  |
| Ceval  | 0.2683       | 0.1463    | 0.4146     | 0.439   | **0.7073** |
| CMB    | 0.155        | 0.08      | 0.27       | 0.275   | **0.765**  |
| CMMLU  | 0.2084       | 0.1315    | 0.379      | 0.3343  | **0.7902** |
| CMExam | 0.185        | 0.145     | 0.35       | 0.26    | **0.73**   |

### 2. 与其他模型的对比结果

| QA-Rouge     |           | SFMed vs. Qwen2-7B-Instruct    | SFMed vs. ChatGPT           | SFMed vs. GPT-4             |
| ------------ | --------- | ------------------------------ | --------------------------- | --------------------------- |
| 多轮对话     | CMtMedQA  | **0.8124/0.0019/0.1857** | 0.3424/0.0019/0.6557        | **0.6538/0.0/0.3462** |
| 单轮对话     | All       | **0.71/0.002/0.288**     | 0.451/0.01/0.539            | **0.54/0.007/0.453**  |
|              | huatuo26M | **0.754/0.002/0.244**    | 0.416/0.008/0.576           | **0.55/0.006/0.444**  |
|              | webMedQA  | **0.666/0.002/0.332**    | 0.486/0.012/0.502           | **0.53/0.008/0.462**  |
| 医学名词解释 | medtiku   | **0.94/0.0/0.06**        | **0.605/0.006/0.389** | **0.842/0.0/0.158**   |

| QA-GPT   |           | SFMed vs. Qwen2-7B-Instruct | SFMed vs. ChatGPT             | SFMed vs. GPT-4      |
| -------- | --------- | --------------------------- | ----------------------------- | -------------------- |
| 多轮对话 | CMtMedQA  | 0.089/0.5841/0.3269         | **0.265/0.5861/0.1489** | 0.1644/0.5184/0.3172 |
| 单轮对话 | All       | 0.087/0.593/0.316           | **0.325/0.554/0.117**   | 0.037/0.505/0.452    |
|          | huatuo26M | 0.11/0.568/0.322            | **0.364/0.506/0.13**    | 0.056/0.516/0.426    |
|          | webMedQA  | 0.064/0.618/0.31            | **0.286/0.602/0.104**   | 0.018/0.494/0.478    |

| 选择题 | ChatGPT | gpt-4  | qwen2-7b-instruct | SFMed            |
| ------ | ------- | ------ | ----------------- | ---------------- |
| All    | 0.4875  | 0.7072 | 0.7153            | **0.7596** |
| PLE    | 0.405   | 0.685  | 0.61              | **0.685**  |
| Ceval  | 0.561   | 0.7317 | **0.7561**  | 0.7073           |
| CMB    | 0.49    | 0.68   | 0.72              | **0.765**  |
| CMMLU  | 0.5021  | 0.7287 | 0.7483            | **0.7902** |
| CMExam | 0.5     | 0.675  | 0.69              | **0.73**   |

## 免责声明 #todo

SFMed信息可能有误，蚂蚁密算科技不保证其准确性、可靠性等，用户需自行承担使用结果和决策责任；蚂蚁密算科技不承担因第三方原因造成的损害责任。SFMed输出的内容不代表蚂蚁密算科技的立场，蚂蚁密算科技不对模型回答承担责任，用户应基于个人判断使用信息，自负医学风险。

## 许可证 #todo

1. 本项目授权协议为 Apache License 2.0，模型权重需要遵守基础模型[Qwen2-7B-Instruct](https://github.com/QwenLM/Qwen2)相关协议及[许可证](https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/LICENSE)，详细内容参照其网站。
2. 使用本项目包括模型权重时请引用本项目：https://github.com/

## 致谢 🎊 #todo

本项目由蚂蚁集团-蚂蚁密算科技发起，负责同学有黄炜、张兆、王莹桂，同时感谢提供的宝贵数据和算力资源。

- 感谢 [hiyouga](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 提供的大模型微调框架。
- 感谢 [SWIFT](https://github.com/modelscope/ms-swift) 提供的 vllm推理加速框架。
- 本项目基于 [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)。

## 引用 #todo

如果您使用或扩展我们的工作，请使用如下的引用格式

```bibtex
@misc{SFMed,
  title = {SFMed},
  author = {},
  howpublished = {\url{https://github.com/}},
  year = {2024}
```

## 联系我们

邮箱：

- [yinggui.wyg@antgroup.com](mailto:yinggui.wyg@antgroup.com)
- [hw378176@antgroup.com](mailto:hw378176@antgroup.com)
- [quanjun.zz@antgroup.com](mailto:quanjun.zz@antgroup.com)

