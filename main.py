# coding=utf-8
import pickle
from random import shuffle
import random
from ToolUtil import evaluate, save, definition_system_prompt, critic_system_prompt, load_example_score, load
from Agent.GPTAgent import GPTAgent
from Constant import *
from baselines.GDEX import GDEX
from baselines.Baseline_Prompts import JsonFormater

random.seed(10086)
import sys

sys.path.append("./metrics")
from dataUtil import load_example, get_stat

print_prompt = False

init_prompt = """
分析一个词在句子中的含义可以从哪些角度出发，以json数组的形式返回结果：[{"角度": [NAME], "角度说明": [STRING]}]
"""

init_points_child = {
    "意图理解": "理解说话者使用该词语的意图和目的，例如说话者是想描述一个物体，还是表达一种情感。",
    "概念形成": "将词语与特定的概念联系起来，例如将“狗”这个词与具有特定特征的动物类别联系起来。",
    "语法理解": "理解词语在句子中的语法角色和功能，例如词语是名词、动词还是形容词，以及它与其他词语之间的关系。",
    "基本学习和记忆": "从该词语的发音和拼写发出，建立它与相关概念之间的联系。",
    "社会线索": "利用说话者的表情、语气、姿势等社会线索来理解词语的含义。",
    "上下文": "词语出现的具体语境，包括前后文和对话背景等。",
}

print("Go!")


def get_definition_by_perspective(definition_agent, word, examples, perspective):
    assert isinstance(examples, list), "wrong type of examples, required list"
    ordered_examples = "\n".join([str(i + 1) + ". " + examples[i] for i in range(len(examples))])
    res = {}

    for pers in perspective:
        pers_values = perspective[pers]
        prompt = """
        根据以下所有[例句]，分析词语[【【】】]的含义，将其总结成一句通顺且易理解的定义，并简要解释原因。
        注意：
        1. 用中文回答
        2. 你需要从[【【】】]角度一步一步地思考这个词语的定义，这意味着去理解【【】】
        3. 在观察用法示例时，要彻底解释上下文，以推断短语的微妙含义。将你的推理分解为循序渐进的逻辑，以达成全面的理解
        4. 你不能过度解读这个词
        5. 以Json形式返回结果：{"词语": "【【】】", "定义": STRING, "原因"：STRING}

        [生成示例]:
        词语: 白人饭
        定义: 白人饭表示一种源自欧美，以生冷食材为主，无需过多加工，高纤维、高蛋白、低热量，且味道较为寡淡的简餐。它经常被用于表示对这种食物的抗拒。这个短语常带有一定的戏谑意味。
        
        ==================
        [例句]:
        """ + ordered_examples
        prompt = JsonFormater().set_string(prompt).format(word, pers, pers_values, word)
        res_ = definition_agent.query(prompt, print_prompt=print_prompt)
        res[pers] = {"definition": res_["定义"], "reason": res_["原因"]}

    return res


def format_def(definition):
    s = ""
    keys = list(definition.keys())
    for i in range(len(keys)):
        key = keys[i]
        defi = definition.get(key)["definition"]
        reas = definition.get(key)["reason"]
        s += str(i) + ". 角度[" + key + "]：" + defi + "。" + reas
        s += "\n"
    return s


def get_critic(critic_agent, word, definition, critic_examples):
    assert isinstance(critic_examples, list), "wrong type of examples, required list"
    assert isinstance(definition, dict), "wrong type of definition, required dict"
    ordered_examples = "\n".join([str(i + 1) + ". " + critic_examples[i] for i in range(len(critic_examples))])

    prompt = """
    根据以下词语[【【】】]、【例句】和【定义】，判断【定义】是否与【例句】中的词义相符。
    如果【定义】与【例句】中的词义不完全一致，你需要总结出一个新的角度，以帮助优化该词【定义】，并说明理由。
    以Json形式返回结果：{"角度": NOUN, "角度解释": STRING, "理由": STRING}
    
    注意
    1. 用中文回答。
    2. 你只能提出一个新角度！
    3. 新角度不能是已经在【定义】中已存在的！
    3. 如果定义与例句中的词义完全一致，则返回：{"角度": "无", "角度解释": [], "理由": []}。
    
    ==============
    【定义】:
    """ + format_def(definition) + """
    
    【例句】:
    """ + ordered_examples + """
    ==============
    【案例】：
    词语：图古德；
    定义：图古德表示一种平静的心情
    例句：这都完成了，这也图古德了吧哈哈哈哈
    
    案例中，定义与例句中的词义不完全一致，返回：{"角度": "谐音", "角度解释": "分析该词的谐音，该词的含义可能体现在其谐音中", "理由": "图古德可能是too good的音译，符合例句里这人”哈哈哈哈“的高兴情绪"}。
    """
    prompt = JsonFormater().set_string(prompt).format(word)
    res = critic_agent.query(prompt, print_prompt=print_prompt)
    print(res)

    if res["角度"] == "无" or res["角度"] == "":
        return "", "", ""
    else:
        return res["角度"], res["角度解释"], res["理由"]


def waus_sentence_selection(examples, top_k):
    return examples


def LLM_sentence_selection(examples, top_k):
    return examples


def summarize_definition(definition_agent, word, examples, reference_definition):
    assert isinstance(examples, list), "wrong type of examples, required list"
    ordered_examples = "\n".join([str(i + 1) + ". " + examples[i] for i in range(len(examples))])

    prompt = """
    根据以下所有[例句]，分析词语[【【】】]的含义，总结该词的[参考定义]成通顺且易理解的定义，包括但不限于本义、引申义和用法等等，并简要解释原因。
    注意：
    1. 用中文回答
    2. 你需要根据[例句]一步一步分析该词[参考定义]的重要性，不是所有的[参考定义]都是有价值的。
    3. 在分析时，要结合[例句]和[参考定义]，以推断[参考定义]的微妙含义，以达成全面的理解。
    4. 以Json形式返回结果：{"词语": "【【】】", "定义": STRING, "原因"：STRING}

    [生成示例]:
    词语: 白人饭
    定义: 白人饭表示一种源自欧美，以生冷食材为主，无需过多加工，高纤维、高蛋白、低热量，且味道较为寡淡的简餐。它经常被用于表示对这种食物的抗拒。这个短语常带有一定的戏谑意味。
    
    ==================    
    [参考定义]:
    """ + format_def(reference_definition) + """
    
    [例句]:
    """ + ordered_examples
    prompt = JsonFormater().set_string(prompt).format(word, word)
    res_ = definition_agent.query(prompt, print_prompt=print_prompt)

    # {"definition": explanation, "perspective_definition": [{perspective: definition}]}
    return {"definition": res_["定义"], "perspective_definition": reference_definition, "reason": res_["原因"]}


# set times_critic = -1 for evaluation using learned Points
def understand_buzzword(definition_agent, critic_agent, word, examples, init_perspective, critic_ratio=0.2,
                        times_critic=3, mark=False):
    # split the examples for critic usage, but we still use all the samples for definition in the end
    assert isinstance(examples, list) and len(
        examples) >= MIN_NUM_EXAMPLE_PER_WORD, "type and number error, need type list, got {}. need example number at least {}, got {}".format(
        str(type(examples)), str(MIN_NUM_EXAMPLE_PER_WORD), str(len(examples)))

    shuffle(examples)
    num_critics = max(1, int(len(examples) * critic_ratio))
    definition_examples = examples  # [num_critics:]

    # make init definition
    # return {perspective: definition}
    res_def = get_definition_by_perspective(definition_agent, word=word, examples=definition_examples,
                                            perspective=init_perspective)
    print("=== init definition ===")
    print(res_def)

    # critic refinement
    i = 0
    critic_info = []
    while i < times_critic:
        critic_examples = random.sample(examples, num_critics)
        # return {perspective: explanation}
        tmp_perspective, tmp_explanation, tmp_reason = get_critic(critic_agent, word=word, definition=res_def,
                                                                  critic_examples=critic_examples)

        if tmp_perspective not in init_perspective and tmp_perspective != "":
            new_def = get_definition_by_perspective(definition_agent, word=word, examples=definition_examples,
                                                    perspective={tmp_perspective: tmp_explanation})
            init_perspective.update({tmp_perspective: tmp_explanation})
            critic_info.append({tmp_perspective: tmp_explanation, "reason": tmp_reason})
            res_def.update(new_def)
            print("=== after {}th critic ===".format(str(i)))
            print(new_def)
        i += 1

    # make final definition
    # {"definition": explanation, "perspective_definition": [{perspective: definition}]}
    res_def = summarize_definition(definition_agent, word=word, examples=definition_examples,
                                   reference_definition=res_def)

    print("========== final ============")
    print(res_def)
    return res_def, init_perspective, critic_info


def main(variants):
    header = "TIGRESS"
    is_mark = False

    # parameters
    sentence_method = variants['sentence_method']
    backbone_method = variants['backbone_method']
    first_k = variants["top_k"]
    critic_nums = 0
    critic_ratio = 0.2 if sentence_method == SENTENCE_METHOD_ALL else 0.5  # how many sentences used for critics
    top_k = "all" if sentence_method == SENTENCE_METHOD_ALL else first_k

    # load agent
    definition_agent = None
    critic_agent = None
    if backbone_method in [GPT_GPT4O_MINI_NAME, GPT_GPT4O_NAME]:
        definition_agent = GPTAgent(model_name=backbone_method, system_prompt=definition_system_prompt
                                    , time_sleep_max=1, time_sleep_min=0)
        critic_agent = GPTAgent(model_name=backbone_method, system_prompt=critic_system_prompt
                                , time_sleep_max=1, time_sleep_min=0)
    elif backbone_method in [QWEN_AGENT_7B_NAME, QWEN_AGENT_72B_NAME]:
        pass
    else:
        raise NotImplementedError
    backbone_name = definition_agent.model_name.replace(" ", "").replace("/", "_").replace("\\", "_")

    # load data and run sentence
    if backbone_method == GPT_GPT4O_NAME:
        from data.key_evaluation_results import gpt_4o

        contamination_label = gpt_4o
    elif backbone_method == GPT_GPT4O_MINI_NAME:
        from data.key_evaluation_results import gpt_4o_mini

        contamination_label = gpt_4o_mini
    elif backbone_method == QWEN_AGENT_7B_NAME:
        from data.key_evaluation_results import qwen_7b

        contamination_label = qwen_7b
    elif backbone_method == QWEN_AGENT_72B_NAME:
        from data.key_evaluation_results import qwen_72b

        contamination_label = qwen_72b
    else:
        contamination_label = None

    if sentence_method == SENTENCE_METHOD_WAUS:
        bert_score_path = "classifier_and_its_attachments/classifier/example_score.csv"
        waus_map = load_example_score(bert_score_path)
    elif sentence_method == SENTENCE_METHOD_GDEX:
        gdex = load("./data/gdex_score.pickle")

    from data.data_cheating_filtered import dataclear

    database = {

    }

    for key in dataclear.keys():
       if key in database:
           continue
       else:
           database[key] = {}
       database[key]["definition"] = ""
       database[key]["ground_truth"] = dataclear[key]['definition']
       database[key]["is_contamination_free"] = 0 if contamination_label[key]["准确性"][0] > 2 or \
                                                     contamination_label[key]["细节完整性"][0] > 2 else 1
       if sentence_method == SENTENCE_METHOD_RANDOM:
           ex = dataclear[key]["examples"]
           s = random.sample(ex, min(top_k, len(ex)))
           database[key]["examples"] = s
       elif sentence_method == SENTENCE_METHOD_ALL:
           database[key]["examples"] = dataclear[key]["examples"]
       elif sentence_method == SENTENCE_METHOD_LLM:
           database[key]["examples"] = LLM_sentence_selection(dataclear[key]["examples"], top_k=top_k)
       elif sentence_method == SENTENCE_METHOD_WAUS:
           database[key]["examples"] = waus_map[key]
       elif sentence_method == SENTENCE_METHOD_GDEX:
           database[key]["examples"] = [i[0] for i in
                                        list(sorted(gdex[key], key=lambda tup: tup[1], reverse=True))[: top_k]]

    get_stat(database)

    # run LLM prompting
    for word in list(database.keys()):
       if "predicted_definition" in database[word] and len(database[word]["predicted_definition"]) > 0:
           continue
       examples = database[word]["examples"]
       definition, init_perspective, critic_info = understand_buzzword(definition_agent, critic_agent, word=word,
                                                                       examples=examples,
                                                                       init_perspective=init_points_child.copy(),
                                                                       times_critic=critic_nums,
                                                                       critic_ratio=critic_ratio,
                                                                       mark=is_mark)
       database[word]["predicted_definition"] = definition["definition"]
       database[word]["definition_reason"] = definition["reason"]
       database[word]["perspective_definition"] = definition["perspective_definition"]
       database[word]["perspective"] = init_perspective
       database[word]["critic_info"] = critic_info
       database[word]["examples"] = ""
       print("【】" + str(len(init_perspective)) + "-+-+" + word + "-+-+" + str(database[word]))

    save(data=database,
        data_path="./data/results/{}_{}_sentence_{}_critic_nums_{}_critic_ratio_{}_wo_evaluation.pickle".format(header,
                                                                                                                backbone_name,
                                                                                                                sentence_method + str(
                                                                                                                    top_k),
                                                                                                                str(critic_nums),
                                                                                                                str(critic_ratio)))

    tmp = { }
    # evaluation
    test_data = evaluate(database, temp_res=tmp)
    save(data=test_data,
         data_path="./data/results/{}_{}_sentence_{}_critic_nums_{}_critic_ratio_{}_w_evaluation.pickle".format(header,
                                                                                                                backbone_name,
                                                                                                                sentence_method + str(
                                                                                                                    top_k),
                                                                                                                str(critic_nums),
                                                                                                                str(critic_ratio)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_method", type=str, default=SENTENCE_METHOD_ALL)
    parser.add_argument("--backbone_method", type=str, default=GPT_GPT4O_NAME)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()
    get_data(vars(args))
