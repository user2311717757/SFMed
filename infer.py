import os
import json
import argparse
from datetime import datetime

from swift.llm import (
    ModelType,
    get_vllm_engine,
    get_default_template_type,
    get_template,
    inference_vllm,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="qwen/Qwen2-7B-Instruct",
    help="model path",
)

args = parser.parse_args()

model_type = ModelType.qwen2.5_7b_instruct
llm_engine = get_vllm_engine(
    model_type,
    model_id_or_path=args.model_path,
    gpu_memory_utilization=0.95,
    tensor_parallel_size=1,
    engine_kwargs={"distributed_executor_backend": "ray"},
)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
llm_engine.generation_config.max_new_tokens = 512
llm_engine.generation_config.temperature = 1.0
print(llm_engine.generation_config)


def inference(llm_engine, template, data_infer):
    return inference_vllm(
        # llm_engine, template, data_infer, use_tqdm=False, verbose=True,
        llm_engine,
        template,
        data_infer,
        use_tqdm=True,
        verbose=False,
    )

PROMPT_multi = "请你基于患者当前及历史的问题给出回复，说话方式要像医生，在必要时如果无法明确诊断患者的疾病，可以询问患者更多的信息。但请切记，不要重复之前轮次的询问。\n"
PROMPT_single = "请你基于患者的问题给出回复，说话方式要像医生。\n"
PROMPT_CHOICE = "请你解题并给出思考过程和答案。【题型】：单选题\n【问题】："

DATA_DICT = {}
DATA_FOLDER = "evaluation_data"
for file in sorted(os.listdir(DATA_FOLDER)):
    with open(os.path.join(DATA_FOLDER, file), "r", encoding="utf-8") as f:
        DATA_DICT[file.split(".")[0]] = [json.loads(i) for i in f.readlines()]

NOW_TIME = datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")
os.makedirs(os.path.join("now_models", NOW_TIME), exist_ok=True)

for key, data_type in DATA_DICT.items():

    if key == "single_choice":
        for i in range(len(data_type)):
            DATA_DICT[key][i]["data"]["query"] = (
                # PROMPT_CHOICE + DATA_DICT[key][i]["data"]["query"]
                PROMPT_CHOICE
                + DATA_DICT[key][i]["data"]["query"].split("\n", 1)[0]
                + "\n【选项】：["
                + DATA_DICT[key][i]["data"]["query"].split("\n", 1)[1]
                + "]"
            )
    elif key == "noun_explanation":
        for i in range(len(data_type)):
            DATA_DICT[key][i]["data"]["query"] = DATA_DICT[key][i]["data"]["query"]
    elif key == "single_turn_qa":
        for i in range(len(data_type)):
            DATA_DICT[key][i]["data"]["query"] = (
                PROMPT_single + DATA_DICT[key][i]["data"]["query"]
            )
    elif key == "multi_turn_qa":
        for i in range(len(data_type)):
            if len(DATA_DICT[key][i]["data"]["history"]) == 0:
                DATA_DICT[key][i]["data"]["query"] = (
                    PROMPT_multi + DATA_DICT[key][i]["data"]["query"]
                )
            else:
                DATA_DICT[key][i]["data"]["history"][0][0] = (
                    PROMPT_multi + DATA_DICT[key][i]["data"]["history"][0][0]
                )
    else:
        print("Unrecognized key")
        continue

    response = inference(llm_engine, template, [i["data"] for i in DATA_DICT[key]])

    for i, res in enumerate(response):
        DATA_DICT[key][i]["response"] = res["response"]

    with open(
        os.path.join("now_models", NOW_TIME, key + ".jsonl"), "w", encoding="utf-8"
    ) as f:
        for d in DATA_DICT[key]:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
