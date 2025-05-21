from data.data import rawdata
from Agent.GeminiAgent import GeminiAgent
from Constant import *
import csv, pickle
import re

url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# xiaohongshu
def get_clear_data(examples_file, examples_clear_file, flag="xhs"):
    line = []
    with open(examples_file, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if flag == "xhs":
                title = row[2].replace("\n", "。").lower().strip()
                desc = row[3].replace("\n", "。").lower().strip()
                if len(title) > 0:
                    title = url_pattern.sub('', title)
                    line.append(title)
                if len(desc) > 0:
                    desc = url_pattern.sub('', desc)
                    line.append(desc)
            elif flag == "wb":
                desc = row[1].replace("\n", "。").lower().strip()
                if len(desc) > 0:
                    desc = url_pattern.sub('', desc)
                    desc = desc.split("//")
                    line += desc

    with open(examples_clear_file, 'w', newline='', encoding='utf-8') as f:
        for l in line:
            f.write(l)
            f.write('\n')

def contains_substring(text, target):
    for i in range(len(target)):
        for j in range(i, len(target)):
            substring = target[i:j+1]
            if len(substring) < 3:
                continue
            if substring in text:
                return True
    return False

def load_example(file_path, rawdata, save_path):
    database = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            database.append(line.lower().strip().strip("。").strip("\t"))
    database = list(set(database))

    # add example using Exact Match
    for k in rawdata.keys():
        examples = [d for d in database if k.lower() in d]
        if len(examples) > 0:
            rawdata[k]['examples'] += examples

    # add example using partial match
    # for k in rawdata.keys():
    #     if len(rawdata[k]['examples']) > MIN_NUM_EXAMPLE_PER_WORD:
    #         continue
    #     examples = [d for d in database if contains_substring(d, k)]
    #     if len(examples) > 0:
    #         rawdata[k]['examples'] += examples

    # stat
    # frequency = Counter([len(rawdata[k]['examples']) for k in rawdata.keys()])
    # for value, count in sorted(frequency.items()):
    #     print(f"{value}, {count}, {value}\t{count}")

    rawdata = clear_data(rawdata)

    save_to_py(rawdata, save_path)

    return rawdata

# 去除没有定义或例句的
def clear_data(data):
    rawdata2 = {k: v for k, v in data.items()
                if len(v['examples']) >= MIN_NUM_EXAMPLE_PER_WORD
                and len(v['ground_truth']) > 0}
    print([k for k in data.keys() if len(data[k]['examples']) < MIN_NUM_EXAMPLE_PER_WORD])
    return rawdata2

def save(data, data_path):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

def save_to_py(data, data_path):
    if os.path.exists(data_path):
        print(data_path, " already exists")
        exit(-1)
    with open(data_path, 'w', encoding="utf-8") as f:
        f.write("rawdata_full = {")
        f.write("\n")
        for k, v in data.items():
            f.write(str({k: v})[1: -1] + ",")
            f.write("\n")
        f.write("}")

def clear_definition():
    system_p = "你是一个词典编撰员，擅长给词下定义。"
    definition_agent = GeminiAgent(model_name=GEMINI_15_PRO_NAME, system_prompt=system_p
                                   , time_sleep_max=60, time_sleep_min=30)

    for k in rawdata.keys():
        desc = rawdata[k]['ground_truth']
        if len(desc) == 0 or "definition" in rawdata[k]:
            continue

        prompt = """
                  以下内容是对一个词语的介绍，你的任务是去理解这段话的内容，然后总结这个词的定义，包括原意与引申意。
                  注意，
                  1. 你给出的定义需要是简洁易懂的一句或多句话
                  2. 以json形式返回结果：{'word': [STRING], 'definition': [STRING]}
                  
                  词语：""" + k + """
                  介绍：""" + desc
        res = definition_agent.query_pool(prompt, GEMINI_KEY_POOL)
        print(res)
        rawdata[k]['definition'] = res['definition']
    save(rawdata, "./data/rawdata_with_definition.pickle")
    return rawdata

def get_stat(data):
    num = len(data) * 1.0
    avg_example = sum([len(data[k]['examples']) for k in data.keys()]) / num
    avg_l_definition = sum([len(data[k]['definition']) for k in data.keys()]) / num
    avg_l_description = sum([len(data[k]['ground_truth']) for k in data.keys()]) / num
    se = [data[k]['examples'] for k in data.keys()]
    avg_l_example = sum([len(x) for xs in se for x in xs]) / (len([x for xs in se for x in xs]) * 1.0)
    print("num, avg_example, avg_l_definition, avg_l_description, avg_l_example: ")
    print(num, avg_example, avg_l_definition, avg_l_description, avg_l_example)

# 去除大模型已经知道含义的
# TODO, 后面让他全跑，为数据打上标签
def word_definition():
    system_p = "你是一个词典编撰员，擅长给词下定义。"
    definition_agent = GeminiAgent(model_name=GEMINI_15_PRO_NAME, system_prompt=system_p
                                   , time_sleep_max=60, time_sleep_min=30)

    for k in rawdata.keys():
        prompt = """
                  给出以下互联网流行词或短语的定义。
                  注意，
                  1. 你给出的定义需要是简洁易懂的一句或多句话
                  2. 以json形式返回结果：{'word': [STRING], 'definition': [STRING]}
                  
                  词语：""" + k
        res = definition_agent.query_pool(prompt, GEMINI_KEY_POOL)
        print(res)
        rawdata[k]['definition'] = res['definition']
    save(rawdata, "./data/rawdata_with_definition.pickle")
    return rawdata

def merge_file(flist, outp):
    with open(outp, 'w', encoding="utf-8") as outfile:
        for fname in flist:
            with open(fname, 'r', encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

if __name__ == '__main__':
    # examples_file = "data/xiaohongshu.csv"
    examples_clear_file = "data/UGC/xiaohongshu_clear.txt"
    # get_clear_data(examples_file, examples_clear_file, flag="xhs")

    # examples_file_wb = "data/weibo.csv"
    examples_clear_file_wb = "data/UGC/weibo_clear.txt"
    # get_clear_data(examples_file_wb, examples_clear_file_wb, flag="wb")

    examples_clear_file_tot = "data/UGC/examples.txt"
    # merge_file([examples_clear_file_wb, examples_clear_file], examples_clear_file_tot)

    save_path = "data/tmp_data/data_full.py"
    # load_example(examples_clear_file_tot, rawdata, save_path)
    # clear_definition()

    from data.tmp_data.data_full import rawdata_full
    get_stat(rawdata_full)

#