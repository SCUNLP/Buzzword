import collections, functools, operator
import sys
sys.path.append("./metrics")
from metrics.bleu.bleu import Bleu
from metrics.rouge.rouge import Rouge
from metrics.bertscore.bertscore import BERTScore
from Agent.GPTAgent import GPTAgent
from Constant import *
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh
import pickle
import csv

##############TOOLS##################
def load_example_score(score_p, top_k=10):
    res = {}
    with open(score_p, "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            word, example, label = line
            if word not in res:
                res[word] = [[example, label]]
            else:
                res[word].append([example, label])
    for k in res.keys():
        res[k] = [i[0] for i in list(sorted(res[k], key=lambda tup: tup[1], reverse=True))[: top_k]]
    return res

def split_data(rawdata2, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
    assert train_ratio + valid_ratio + test_ratio == 1, "wrong ratio"
    print("reading data ", len(rawdata2))
    all_index = list(rawdata2.keys())
    shuffle(all_index)
    train_index = all_index[: int(train_ratio * len(all_index))]
    valid_index = all_index[int(train_ratio * len(all_index)): int(train_ratio * len(all_index)) + int(valid_ratio * len(all_index))]
    test_index = all_index[int(train_ratio * len(all_index)) + int(valid_ratio * len(all_index)): ]

    return {k: v for k, v in rawdata2.items() if k in train_index}, \
        {k: v for k, v in rawdata2.items() if k in valid_index}, \
        {k: v for k, v in rawdata2.items() if k in test_index}


definition_system_prompt = "你是一个资深的词典编撰专家，擅长分析给定词语在若干案例中的含义，并为该词下定义。"
critic_system_prompt = "你是一个资深的语言学家，是词典编撰专家的助手，擅长验证给定词语含义与例句中实际使用的含义是否一致，并为词典编撰专家提供修改建议。"
evaluation_system_prompt = "你是一个资深的词典编撰专家，擅长评估词语定义的质量"

bleu_metric = Bleu() #ev.load(path="./metrics/bleu/bleu.py")
rouge_metric = Rouge() #ev.load(path="./metrics/rouge/rouge.py")
bert_score_metric = BERTScore()
a = TokenizerZh()
print("loaded metrics")


##############TOOLS#################

def save(data, data_path):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


def load(data_path):
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def calculate_rouge(predictions, groud_truth):
    predictions = [predictions] if not isinstance(predictions, list) else predictions
    references = [groud_truth] if not isinstance(groud_truth, list) else groud_truth
    results = rouge_metric.compute(predictions=predictions, references=references, tokenizer=a)
    # Extract all ROUGE scores
    rouge_scores = {
        f"rouge1_{k}": v for k, v in results.items() if k.startswith('rouge1')
    }
    rouge_scores.update({
        f"rouge2_{k}": v for k, v in results.items() if k.startswith('rouge2')
    })
    rouge_scores.update({
        f"rougeL_{k}": v for k, v in results.items() if k.startswith('rougeL')
    })
    rouge_scores.update({
        f"rougeLsum_{k}": v for k, v in results.items() if k.startswith('rougeLsum')
    })
    return rouge_scores


def calculate_bleu(predictions, groud_truth):
    predictions = [predictions] if not isinstance(predictions, list) else predictions
    references = [groud_truth] if not isinstance(groud_truth, list) else groud_truth
    results = bleu_metric.compute(predictions=predictions, references=references, tokenizer=a)
    return results


def calculate_bertscore(predictions, groud_truth):
    predictions = [predictions] if not isinstance(predictions, list) else predictions
    references = [groud_truth] if not isinstance(groud_truth, list) else groud_truth
    results = bert_score_metric.compute(predictions=predictions, references=references, lang='zh')
    return results

##############FUNCTION###############


def get_evaluation(definition, ground_truth, evaluation_agent):
    prompt = """
    给定一个词语的【定义】，你需要从以下【评估角度和打分标准】，为这个【定义】的质量高低评分。
    使用Json格式返回结果：{"简洁可读性": [INT, WHY], "客观性": [INT, WHY], "适用范围": [INT, WHY]}
    
    
    【定义】：""" + definition + """
    
    【评估角度和打分标准】：
    ===================
    简洁可读性：
    1分：定义冗长、模糊不清，或者使用了复杂难懂的词汇或者语法结构而难以阅读和理解。
    2分：定义较为简洁但可能仍含有不必要的词汇复杂性、结构复杂性、或者缺乏清晰度，可读性有待提高。
    3分：定义清晰简洁，但用词上仍有改进空间，可读性一般。
    4分：定义既清晰又简洁，使用了恰当的语言和结构，易于阅读。
    5分：定义非常准确、清晰且简洁，使用了简单易懂的语言，非常易于阅读。
    
    客观性：
    1分：定义明显带有个人偏见或情感色彩。
    2分：定义中存在一定的主观性。
    3分：定义比较客观，但偶尔出现轻微的倾向性。
    4分：定义客观，未表现出明显的偏好。
    5分：定义完全客观，没有任何偏见。
    
    适用范围：
    1分：未说明词语的适用范围。
    2分：略微提及了词语的适用范围。
    3分：基本指出了词语的适用范围，但不够具体。
    4分：明确指出了词语的适用范围，包括语体色彩等。
    5分：详细指出了词语的适用范围，包括地区、领域等。

    ===================
    """

    prompt2 = """
    给定一个词语的【定义】和专家给出的【参考定义】，你需要从以下【评估角度和打分标准】，为这个【定义】的质量高低评分。
    使用Json格式返回结果：{"准确性": [INT, WHY], "细节完整性": [INT, WHY]}
    
    
    【定义】：""" + definition + """
    【参考定义】：""" + ground_truth + """
    
    【评估角度和打分标准】：
    ===================
    准确性：
    1分：该定义与【参考定义】相比，严重偏离了词语的真实意义，或者包含大量错误信息。
    2分：该定义与【参考定义】相比，有一定的偏差，但至少部分正确。
    3分：该定义与【参考定义】相比，基本准确，但可能存在一些小错误或不完整的描述。
    4分：该定义与【参考定义】相比，准确，能够清晰地传达词语的核心意义。
    5分：该定义与【参考定义】相比，非常准确，全面反映了词语的意义，没有遗漏重要细节。
    
    细节完整性：
    1分：该定义与【参考定义】相比，遗漏了许多重要的细节。
    2分：该定义与【参考定义】相比，遗漏了一些重要的细节，但整体还算完整。
    3分：该定义与【参考定义】相比，包含大部分必要细节，但仍有改进空间。
    4分：该定义与【参考定义】相比，包含了几乎所有必要的细节。
    5分：该定义与【参考定义】相比，包含了所有必要的细节，没有遗漏。
    ===================
    """

    res = evaluation_agent.query(prompt, print_prompt=False)
    res2 = evaluation_agent.query(prompt2, print_prompt=False)
    if isinstance(res, str):
        print(res)
        res = {"简洁可读性": [1, "我无法评价"], "客观性": [1, "我无法评价"], "适用范围": [1, "我无法评价"]}
    res.update(res2)
    return res


def evaluate(data, temp_res={}):
    # temp_res = {
    #
    # }
    # load
    for word in temp_res.keys():
        data[word]["evaluation"] = temp_res.get(word)

    evaluation_agent = GPTAgent(model_name=GPT_GPT4O_NAME, system_prompt=evaluation_system_prompt
                                , time_sleep_max=1, time_sleep_min=0)

    for word in data.keys():
        if word in temp_res:
            continue
        ground_truth = data[word]["ground_truth"]
        predicted_definition = data[word]["predicted_definition"]

        # automatic evaluation via RoughL and BLEU
        res = {}
        rouge = calculate_rouge(predicted_definition, ground_truth)
        bleu = calculate_bleu(predicted_definition, ground_truth)
        bertscore = calculate_bertscore(predicted_definition, ground_truth)

        res["rougeL"] = rouge["rougeL_rougeL"] if not isinstance(rouge["rougeL_rougeL"], list) else rouge["rougeL_rougeL"][0]
        res["bleu"] = bleu["bleu"] if not isinstance(bleu["bleu"], list) else bleu["bleu"][0]
        res["bert_score"] = bertscore["f1"] if not isinstance(bertscore["f1"], list) else bertscore["f1"][0]

        # automatic evaluation via simulated evaluator
        llm_res = get_evaluation(predicted_definition, ground_truth, evaluation_agent)
        res.update(llm_res)
        print(word, str(res))

        data[word]["evaluation"] = res
    print("===== Performance ======")
    tmp_ = list(data.values())
    tmp = [{k: v if not isinstance(v, list) else v[0] for k, v, in tmp["evaluation"].items()} for tmp in tmp_]
    sum_tmp = dict(functools.reduce(operator.add, map(collections.Counter, tmp)))
    avg_tmp = {k: v/len(tmp) for k, v, in sum_tmp.items()}
    print("overall")
    print(avg_tmp)

    tmp = [{k: v if not isinstance(v, list) else v[0] for k, v, in tmp["evaluation"].items()} for tmp in tmp_ if tmp['is_contamination_free'] == 1]
    sum_tmp = dict(functools.reduce(operator.add, map(collections.Counter, tmp)))
    avg_tmp = {k: v/len(tmp) for k, v, in sum_tmp.items()}
    print("is_contamination_free = 1")
    print(avg_tmp)

    tmp = [{k: v if not isinstance(v, list) else v[0] for k, v, in tmp["evaluation"].items()} for tmp in tmp_ if tmp['is_contamination_free'] == 0]
    sum_tmp = dict(functools.reduce(operator.add, map(collections.Counter, tmp)))
    avg_tmp = {k: v/len(tmp) for k, v, in sum_tmp.items()}
    print("is_contamination_free = 0")
    print(avg_tmp)
    return data
