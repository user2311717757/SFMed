# SFMed

SFMed is a large-scale medical domain model that has undergone continued pre-training, supervised fine-tuning, and alignment based on Qwen2.5-7B-Instruct. Its current performance has comprehensively surpassed that of other open-source medical models and is comparable to proprietary models.


The core functions of SFMed include:

- Medical Consultation: It can act as a doctor to answer users' questions about diseases and other health-related issues. This includes single-turn question-and-answer dialogues as well as multi-turn dialogues with follow-up questions.

- Pharmaceutical Consultation: It understands medical terminology, drug names, and other specialized terms, providing precise professional knowledge in the medical field.

## Update Log

[2024/11/01] 🚀[Open-source SFMed and 7B model weights🤗](https://huggingface.co/) #todo (Once  the  paper  is  accepted,  we  will  release  our  model.)

### Download

[SFMed-7B-Instruct](https://huggingface.co/) #todo

### Inference

Same inference method as [🤗Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## Training Data

A total of approximately 2.7B tokens, including around 1.4B tokens of general corpus and about 1.3B tokens from the medical domain.

- Chinese medical data

| Dataset name                    | Dataset description                                   | Number of entries in the dataset | Number of tokens in the dataset |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ------------- |
| CMtMedQA                      | The dataset contains 70,000 multi-turn dialogues derived from real doctor-patient interactions, including many proactive inquiry statements. | 68023      | 25569575      |
| ChatMed_Consult_Dataset       | The medical consultation questions are sourced from the internet, reflecting the diverse medical consultation needs of different users/patients in the real world. Currently, the responses are provided by the OpenAI GPT-3.5 engine.| 549326     | 79788168      |
| DISC-Med-SFT                  | It includes over 470,000 samples derived from the reconstruction of existing medical datasets.      | 464898     | 147775556     |
| MedDiag                       | Chinese dialogues between doctors and patients.                                                             | 2725990    | 351973718     |
| PromptCBLUE                   | Further developed the CBLUE benchmark, transforming all 16 different medical scenario NLP tasks into prompt-based language generation tasks, thus creating the first LLM evaluation benchmark for Chinese medical scenarios.   | 151500     | 27127160      |
| ShenNong_TCM                  | Using the Traditional Chinese Medicine knowledge graph, an entity-centered self-instruct method was used to generate over 110,000 instruction data related to Traditional Chinese Medicine by leveraging ChatGPT.  | 112565     | 25168926      |
| cMedQA-V2.0                   | A dataset of Chinese community medical questions and answers.               | 226266     | 22608102      |
| huatuo_knowledge_graph_qa     | This QA dataset was constructed based on a medical knowledge graph and contains a total of 798,444 entries. The questions were generated using templates, while the answers were derived from the contents of the knowledge graph entries. | 798444     | 30869480      |
| huatuo_sft_train_data         | Part of the data used to train HuatuoGPT                 | 226042     | 67348069      |
| huatuo_encyclopedia_qa        | Medical QA pairs were extracted from medical encyclopedias and medical articles. A total of 8,699 disease encyclopedia entries and 2,736 drug encyclopedia entries were collected from Chinese Wikipedia. Additionally, we scraped 226,432 high-quality medical articles from the Qianwen Health website. | 362420     | 135298899     |
| webMedQA                      | A dataset of real Chinese medical questions and answers collected from online health consultation websites.                                      | 316110     | 56211345      |
| Chinese-medical-dialogue-data | Chinese Medical QA Dataset                                                                    | 792099     | 115022641     |
| CMExam                        | A dataset from the Chinese National Medical Licensing Examination, including over 60K multiple-choice questions.                    | 68119      | 15019251      |
| CMB-Exam                      | Comprehensive and multi-level evaluation of the model's medical knowledge                                                                   | 280839     | 18724518      |
| questions                     | Some Chinese medical data                                                                                         | 48376      | 4376377       |
| Textbook                        | Chinese medical textbook data                                                                                  | 21471      | 15179698      |
| Crawler                       | Data scraped from websites like Haodf and DXY.                                                                          | 482922     | 68760072      |
| yiigle                        | Titles and abstracts of Chinese medical journals                                                                 | 308869     | 104137233     |

- General data in Chinese

| Dataset name    |  Dataset description              | Number of entries in the dataset | Number of tokens in the dataset |
| ------------- | ------------------------------------------------------------------------------------ | ---------- | ------------- |
| baike2018qa   | Encyclopedic Q&A, containing 1.5 million pre-filtered, high-quality questions and answers.                                | 1470142    | 358976186     |
| webtext2019zh | Community Q&A, containing 4.1 million pre-filtered, high-quality questions and responses.                                | 4258310    | 763721567     |
| wiki2019zh    | Wikipedia can be used as general Chinese corpus for pre-training or constructing word vectors, and it can also be used for building knowledge-based Q&A systems. | 1248027    | 314398588     |

## Evaluation

Evaluation dataset：

The evaluation dataset includes multi-turn dialogues, single-turn dialogues, medical term explanations, and multiple-choice question data.

- CMtMedQA：[https://github.com/SupritYoung/Zhongjing/blob/main/data/CMtMedQA_test.json](https://github.com/SupritYoung/Zhongjing/blob/main/data/CMtMedQA_test.json)
- huatuo26M：[https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets](https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets)
- webMedQA：[https://github.com/hejunqing/webMedQA](https://github.com/hejunqing/webMedQA)
- medtiku：[https://www.medtiku.com/](https://www.medtiku.com/)
- Multiple-choice questions: Multiple-choice question data related to the medical field extracted from sources such as CEval.

Evaluation method：

Download the evaluation dataset, replace the model path, and then run the code below.

Dependencies：[SWIFT](https://github.com/modelscope/ms-swift)

```python
python infer_vllm.py --model_path 
```

### 1. Comparison results with authoritative medical open-source models

| QA-Rouge     |           | SFMed vs. HuatuoGPT-II      | SFMed vs. Zhongjing            | SFMed vs. ChiMed-GPT           | SFMed vs. WiNGPT2           |
| ------------ | --------- | --------------------------- | ------------------------------ | ------------------------------ | --------------------------- |
| multi-turn dialogues     | CMtMedQA  | **0.7544/0.0/0.2456** | **0.5919/0.0019/0.4062** | **0.8607/0.0019/0.1373** | 0.3675/0.0/0.6325           |
| single-turn dialogues     | All       | **0.558/0.008/0.434** | **0.539/0.014/0.447**    | **0.5/0.013/0.487**      | **0.545/0.01/0.445**  |
|              | huatuo26M | **0.584/0.008/0.408** | **0.506/0.016/0.478**    | **0.506/0.008/0.486**    | **0.532/0.008/0.46**  |
|              | webMedQA  | **0.532/0.008/0.46**  | **0.572/0.012/0.416**    | **0.494/0.018/0.488**    | **0.558/0.012/0.43**  |
| medical term explanations | medtiku   | **0.76/0.003/0.237**  | **0.638/0.002/0.36**     | **0.687/0.005/0.308**    | **0.654/0.001/0.345** |

| QA-GPT   |           | SFMed vs. HuatuoGPT-II         | SFMed vs. Zhongjing            | SFMed vs. ChiMed-GPT           | SFMed vs. WiNGPT2              |
| -------- | --------- | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------ |
| multi-turn dialogues | CMtMedQA  | **0.3907/0.4874/0.1219** | **0.6209/0.3366/0.0426** | **0.9884/0.0097/0.0019** | **0.6209/0.3133/0.0658** |
| single-turn dialogues | All       | **0.242/0.567/0.186**    | **0.702/0.248/0.045**    | **0.975/0.013/0.005**    | **0.861/0.114/0.019**    |
|          | huatuo26M | **0.282/0.54/0.178**     | **0.712/0.244/0.044**    | **0.972/0.016/0.008**    | **0.876/0.106/0.014**    |
|          | webMedQA  | **0.202/0.594/0.194**    | **0.692/0.252/0.046**    | **0.978/0.01/0.002**     | **0.846/0.122/0.024**    |

| Multiple-choice question | HuatuoGPT-II | Zhongjing | ChiMed-GPT | WiNGPT2 | SFMed            |
| ------ | ------------ | --------- | ---------- | ------- | ---------------- |
| All    | 0.1844       | 0.1195    | 0.3348     | 0.3075  | **0.7596** |
| PLE    | 0.11         | 0.085     | 0.21       | 0.265   | **0.685**  |
| Ceval  | 0.2683       | 0.1463    | 0.4146     | 0.439   | **0.7073** |
| CMB    | 0.155        | 0.08      | 0.27       | 0.275   | **0.765**  |
| CMMLU  | 0.2084       | 0.1315    | 0.379      | 0.3343  | **0.7902** |
| CMExam | 0.185        | 0.145     | 0.35       | 0.26    | **0.73**   |

### 2. Comparison results with other models

| QA-Rouge     |           | SFMed vs. Qwen2.5-7B-Instruct    | SFMed vs. ChatGPT           | SFMed vs. GPT-4             |
| ------------ | --------- | ------------------------------ | --------------------------- | --------------------------- |
| multi-turn dialogues | CMtMedQA  | **0.8124/0.0019/0.1857** | 0.3424/0.0019/0.6557        | **0.6538/0.0/0.3462** |
| single-turn dialogues | All       | **0.71/0.002/0.288**     | 0.451/0.01/0.539            | **0.54/0.007/0.453**  |
|              | huatuo26M | **0.754/0.002/0.244**    | 0.416/0.008/0.576           | **0.55/0.006/0.444**  |
|              | webMedQA  | **0.666/0.002/0.332**    | 0.486/0.012/0.502           | **0.53/0.008/0.462**  |
| medical term explanations| medtiku   | **0.94/0.0/0.06**        | **0.605/0.006/0.389** | **0.842/0.0/0.158**   |

| QA-GPT   |           | SFMed vs. Qwen2.5-7B-Instruct | SFMed vs. ChatGPT             | SFMed vs. GPT-4      |
| -------- | --------- | --------------------------- | ----------------------------- | -------------------- |
| multi-turn dialogues | CMtMedQA  | 0.089/0.5841/0.3269         | **0.265/0.5861/0.1489** | 0.1644/0.5184/0.3172 |
| single-turn dialogues | All       | 0.087/0.593/0.316           | **0.325/0.554/0.117**   | 0.037/0.505/0.452    |
|          | huatuo26M | 0.11/0.568/0.322            | **0.364/0.506/0.13**    | 0.056/0.516/0.426    |
|          | webMedQA  | 0.064/0.618/0.31            | **0.286/0.602/0.104**   | 0.018/0.494/0.478    |

| Multiple-choice question | ChatGPT | gpt-4  | qwen2.5-7b-instruct | SFMed            |
| ------ | ------- | ------ | ----------------- | ---------------- |
| All    | 0.4875  | 0.7072 | 0.7153            | **0.7596** |
| PLE    | 0.405   | 0.685  | 0.61              | **0.685**  |
| Ceval  | 0.561   | 0.7317 | **0.7561**  | 0.7073           |
| CMB    | 0.49    | 0.68   | 0.72              | **0.765**  |
| CMMLU  | 0.5021  | 0.7287 | 0.7483            | **0.7902** |
| CMExam | 0.5     | 0.675  | 0.69              | **0.73**   |
