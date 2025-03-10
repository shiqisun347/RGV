#! /usr/bin/env python
# -- coding: utf-8 --
"""
@author:347
@email:1076598466@qq.com
@file:GenRead-main-5.实体消歧.py
@software:PyCharm
@time:2023/12/9 10:20
"""
from collections import defaultdict

from tqdm import tqdm
from util import *
import openai

if __name__ == "__main__":
    process_data = readfiles("webq_test_step8_12061815_output.jsonl")

    outfile = open("data/webq_test_step9_12061815_output2.jsonl", 'a', encoding='utf8')

    for pd in tqdm(process_data):

        # 处理 子图生成的实体
        sents = [po.strip('\n') for po in pd['output']]
        pd["entities"][0].extend(sents)
        pd["entities"][0] = list(set(pd["entities"][0]))
        for ent in pd["entities"]:
            if ent in pd["answername"]:
                continue

        entities = pd["entities"][0]

        # words = answers
        for word in pd["answername"]:
            for idx, target_word in enumerate(entities):
                # 计算与目标词的相似性
                if target_word in pd["answername"]:
                    continue
                    # 你的文本
                doc1 = word
                doc2 = target_word

                # 将两个文档分词

                words_doc1 = [ds.lower() for ds in split_doc1] if len(split_doc1) > 1 else doc1.lower()
                words_doc2 = [ds.lower() for ds in split_doc2] if len(split_doc2) > 1 else doc2.lower()

                split_doc1 = doc1.split(" ")
                split_doc2 = doc2.split(" ")
                # 均为带空格分割的字符  Jaccard相似度
                if len(split_doc1) > 1 or len(split_doc2) > 1:

                    similarity = jaccard_similarity(words_doc1, words_doc2)

                if len(doc2.split(" ")) == 1 and len(doc2.split(" ")) == 1 and similarity > 0.5:
                    entities[idx] = word
                    print("Jaccard Similarity: ", (words_doc1, words_doc2), similarity)
                if (len(doc2.split(" ")) > 1 or len(doc2.split(" ")) > 1) and similarity > 0.3:
                    entities[idx] = word
                    print("Jaccard Similarity: ", (words_doc1, words_doc2), similarity)
        pd["entities"][0] = entities
        outfile.write(json.dumps(pd) + '\n')
    outfile.close()
