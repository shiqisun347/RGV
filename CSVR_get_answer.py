#! /usr/bin/env python
# -- coding: utf-8 --
"""
@author:347
@email:1076598466@qq.com
@file:GenRead-main-9.根据子图生成答案.py
@software:PyCharm
@time:2023/12/10 20:36
"""
prompt = '''Refer to the following subgraph and answer the current question with only one entity as the answer.\n 
question :{}? \n subgraph :{} \nThe answer is only one entity.'''
from exp_mqa.util import *

if __name__ == "__main__":

    web_test = readfiles("data/webq_test_step7_12061815_output.jsonl")

    outputfile = "webq_test_step8_12061815_output.jsonl"

    if os.path.exists(outputfile):
        outs = open(outputfile, 'a', encoding='utf8')
        num_lines = len(open(outputfile, 'r').readlines())
        web_test = web_test[num_lines - 1:]
    else:  # not os.path.exists(outfile)
        outs = open(outputfile, 'a', encoding='utf8')
        outs.write(json.dumps({"prompt": prompt}) + '\n')

    pbar = tqdm(total=len(web_test))
    index = 0
    pbar.update(index)
    while index < len(web_test):
        inputs, answers = [], []
        inputs_with_prompts = []
        for _ in range(20):
            if index >= len(web_test): break
            input_with_prompt = add_prompt(web_test[index], prompt)

            inputs.append(web_test[index])
            inputs_with_prompts.append(input_with_prompt)
            index += 1
        print(inputs_with_prompts)
        samples = defaultdict(list)
        completions = {"choices": []}
        for _ in range(200):
            try:
                with time_limit(20, 'run gpt-3'):
                    completions = openai.Completion.create(
                        engine='text-davinci-002',
                        max_tokens=2048,
                        prompt=inputs_with_prompts,
                        temperature=0,
                        n=1,  # num of returned sequence
                    )
                    break
            except:
                time.sleep(2)

        outputs = [c["text"] for c in completions["choices"]]

        for j, output in enumerate(outputs):
            samples[j].append(output.strip("\n"))
        for i in range(len(inputs_with_prompts)):
            inputs[i]['output'] = samples[i]
            outs.write(json.dumps(inputs[i]) + '\n')

        pbar.update(len(inputs_with_prompts))

    pbar.close()
    outs.close()

    exact_match_count = 0
    answer_lengths = []
    count = 0
    for line in lines:

        line = json.loads(line)
        tag = False
        output = [key for key, value in line['paths'].items() if 4 > len(value[0]) > 1]

        if len(output) > 0:
            op = output[0]
        else:
            op = ''
            count += 1
        if ems(op, line['answername']):  # EM evaluation
            tag = True

        else:
            print(line)

        answer_lengths.append(len(op.split()))
        exact_match_count += 1 if tag else 0
    emscore = round(exact_match_count / len(lines), 4)
    length = round(np.mean(answer_lengths), 4)
    print(count)
    print(f'Exact Match: {emscore}; Avg.Length: {length}')
