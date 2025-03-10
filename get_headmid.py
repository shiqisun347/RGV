#! /usr/bin/env python
# -- coding: utf-8 --
"""
@author:347
@email:1076598466@qq.com
@file:GenRead-main-获取所有headmid.py
@software:PyCharm
@time:2023/12/10 0:03
"""

from tqdm import tqdm
from util import *
import openai

if __name__ == "__main__":
    raw_data = readfiles("data/web_test.jsonl")
    outfile = open("data/webq_test_step4_12061815.jsonl", 'a', encoding='utf8')
