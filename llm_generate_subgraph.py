#! /usr/bin/env python
# -- coding: utf-8 --
"""
@author:347
@email:1076598466@qq.com
@file:GenRead-main-2.llm_generate_subgraph.py
@software:PyCharm
@time:2023/12/6 19:01
"""
from collections import defaultdict

from tqdm import tqdm
from util import *
import openai



if __name__ == "__main__":
    process_data = readfiles("data/webq_test_step1_12061815.jsonl")

    outfile = open("data/webq_test_step2_12061815.jsonl", 'a', encoding='utf8')



    pbar = tqdm(total=len(process_data))
    index = 0
    pbar.update(index)
    while index < len(process_data):
        inputs = []
        inputs_with_prompts = []
        for _ in range(20):
            if index >= len(process_data): break
            input_with_prompt = process_data[index]["prompts"]
            inputs.append(process_data[index])
            inputs_with_prompts.append(input_with_prompt)
            index += 1

        samples = defaultdict(list)

        outputs = run_inference(inputs_with_prompts, engine="text-davinci-003")
        for j, output in enumerate(outputs):
            samples[j].append(output)

        for i in range(len(inputs_with_prompts)):
            inputs[i]["subgraph"] = samples[i]
            outfile.write(json.dumps(inputs[i]) + '\n')
        pbar.update(len(inputs_with_prompts))

    pbar.close()
    outfile.close()
