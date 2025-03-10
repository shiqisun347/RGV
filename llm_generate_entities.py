#! /usr/bin/env python
# -- coding: utf-8 --
"""
@author:347
@email:1076598466@qq.com
@file:GenRead-main-3.llm_generate_reference.py
@software:PyCharm
@time:2023/12/6 20:31
"""
# ! /usr/bin/env python
# -- coding: utf-8 --

from collections import defaultdict

from tqdm import tqdm
from util import *
import openai

# entities reference subgraph
# task = "generate_entities"

if __name__ == "__main__":

    process_data = readfiles("data/webq_test_step3_12061815.jsonl")

    outfile = open("data/webq_test_step4_12061815.jsonl", 'a', encoding='utf8')

    pbar = tqdm(total=len(process_data))
    index = 0
    pbar.update(index)
    while index < len(process_data):
        inputs = []
        inputs_with_prompts = []
        for _ in range(20):
            if index >= len(process_data): break
            input_with_prompt = ("According to the question, extract the entity list from the reference as alternative "
                                 "answers.\n Question:{}? \n Reference: {} \n Entity list is? Entity name must come "
                                 "from Wikipedia.Use commas to separate entities").format(
                process_data[index]["unmasked_question"], process_data[index]["reference"])
            inputs.append(process_data[index])
            inputs_with_prompts.append(input_with_prompt)
            index += 1
        samples = defaultdict(list)
        outputs = run_inference(inputs_with_prompts, engine="text-davinci-003")
        for j, output in enumerate(outputs):
            samples[j].append(output)

        for i in range(len(inputs_with_prompts)):
            inputs[i]["entities"] = samples[i]
            outfile.write(json.dumps(inputs[i]) + '\n')
        pbar.update(len(inputs_with_prompts))

    pbar.close()
    outfile.close()
