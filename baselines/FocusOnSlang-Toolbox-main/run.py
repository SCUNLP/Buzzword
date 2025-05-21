import random

from agents import *
from tqdm import tqdm
import datetime
#from eval_urban_f1 import calculate_metrics
import re
from Agent.QwenAgent import QwenAgent
from Agent.GPTAgent import GPTAgent
from Agent.GeminiAgent import GeminiAgent
from data_cheating_filtered import get_data

# Urban Dictionary Templates
URBAN_DICTIONARY_TEMPLATES = {
    'Direct': lambda prompt_entity: urban_direct_prompt.format(*prompt_entity),
    'Instruction': lambda prompt_entity: urban_instruction_prompt.format(word=prompt_entity[0], example=prompt_entity[1]),
    'ICL': lambda prompt_entity: urban_icl_prompt.format(word=prompt_entity[0], example=prompt_entity[1]),
    'CoT': lambda prompt_entity: urban_cot_prompt.format(word=prompt_entity[0], example=prompt_entity[1]), # plain guess
    'Causal': lambda prompt_entity: urban_causal_prompt.format(example=prompt_entity[1]), # masked guess
    'GenerateNewSentences': lambda prompt_entity: urban_causal_mid_prompt.format(word=prompt_entity[0], example=prompt_entity[1]), # generate new sentences
    'SummarizeResult': lambda prompt_entity: urban_causal_final_prompt.format(*prompt_entity),
    # Add other policies if needed
    'Causal_Propose': lambda prompt_entity: urban_causal_legacy_propose_prompt.format(word=prompt_entity[0], example=prompt_entity[1]),
    'Causal_CoT': lambda prompt_entity: urban_causal_legacy_cot_prompt.format(phrase=prompt_entity[0],
                                                                              example=prompt_entity[1],
                                                                              reconstructed_example=prompt_entity[2],
                                                                              entity_candidates_json=prompt_entity[3])
}

class CommonAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()

    def execute(self, agent, prompt):
        current_joint_memory = {}
        raw_result = agent.get_response(prompt)
        current_joint_memory['plain_guess'] = agent.short_memory
        return raw_result, current_joint_memory


class CausalAlgorithm(Algorithm):
    def __init__(self, llm):
        super().__init__()
        self.causal_propose_agent = Agent(llm, parser=causal_propose_parser)
        self.causal_cot_agent = Agent(llm, parser=cot_parser)
        self.causal_propose_task = Task(URBAN_DICTIONARY_TEMPLATES['Causal_Propose'])
        self.causal_cot_task = Task(URBAN_DICTIONARY_TEMPLATES['Causal_CoT'])

    def execute(self, word_with_example):
        current_joint_memory = {}
        
        causal_propose_prompt = self.causal_propose_task.get_prompt(word_with_example)
        causal_propose_response = self.causal_propose_agent.get_response(causal_propose_prompt)
        current_joint_memory['causal_propose'] = self.causal_propose_agent.short_memory
        entity_list = causal_propose_response['entity_replacement_list']
        recon_example = causal_propose_response['reconstructed_example']
        print("Causal Propose Response: {}".format(causal_propose_response))
        # time.sleep(1)

        causal_cot_prompt = self.causal_cot_task.get_prompt((word_with_example[0], word_with_example[1], entity_list, recon_example))
        causal_cot_response = self.causal_cot_agent.get_response(causal_cot_prompt)
        current_joint_memory['causal_cot'] = self.causal_cot_agent.short_memory
        print("Causal CoT Response: {}".format(causal_cot_response))
        # time.sleep(1)

        return causal_cot_response.replace('[MASKED_PHRASE]', word_with_example[0]) , current_joint_memory

        
class FOCUSAlgorithm(Algorithm):
    def __init__(self, llm):
        super().__init__()
        self.guess_agent = Agent(llm, parser=cot_parser)
        self.masked_guess_agent = Agent(llm, parser=masked_guess_parser)
        self.generate_agent = Agent(llm, parser=generate_new_sentences_parser)
        self.summarize_agent = Agent(llm, parser=summarize_parser)
        self.plain_guess_task = Task(URBAN_DICTIONARY_TEMPLATES['CoT'])
        self.masked_guess_task = Task(URBAN_DICTIONARY_TEMPLATES['Causal'])
        self.generate_task = Task(URBAN_DICTIONARY_TEMPLATES['GenerateNewSentences'])
        self.generate_guess_task = Task(URBAN_DICTIONARY_TEMPLATES['Causal'])
        self.summarize_task = Task(URBAN_DICTIONARY_TEMPLATES['SummarizeResult'])
        

    def execute(self, word_with_example):

        current_joint_memory = {}
        # Plain guess
        plain_guess_prompt = self.plain_guess_task.get_prompt(word_with_example)
        plain_guess_response = self.guess_agent.get_response(plain_guess_prompt)
        current_joint_memory['plain_guess'] = self.guess_agent.short_memory
        print("Plain Guess Response: {}".format(plain_guess_response))
        # time.sleep(1)

        # # Masked guess
        MASKRD_PHRASE_with_example = (word_with_example[0], replace_ignore_case(word_with_example[1], word_with_example[0], '[MASKRD_PHRASE]'))
        masked_guess_prompt = self.masked_guess_task.get_prompt(MASKRD_PHRASE_with_example)
        masked_guess_response = self.masked_guess_agent.get_response(masked_guess_prompt)
        current_joint_memory['masked_guess'] = self.masked_guess_agent.short_memory
        print("Masked Guess Response: {}".format(masked_guess_response))
        # time.sleep(1)

        # Generate new sentences
        generate_response = None
        masked_sentences = replace_ignore_case(word_with_example[1], word_with_example[0], '[MASKRD_PHRASE]')
        for start in range(0, len(masked_sentences), 20):
            if start + 20 > len(masked_sentences):
                end = len(masked_sentences)
            else:
                end = start + 20
            example = masked_sentences[start:end]
            MASKRD_PHRASE_with_example = (word_with_example[0], example)
            generate_prompt = self.generate_task.get_prompt(MASKRD_PHRASE_with_example)
            part_response = self.generate_agent.get_response(generate_prompt)
            if generate_response is None:
                generate_response = part_response
            else:
                generate_response = generate_response + '\n' + part_response
        current_joint_memory['generate'] = self.generate_agent.short_memory
        print("Generate Response: {}".format(generate_response))

        entity_replaced_prompt = self.generate_guess_task.get_prompt((word_with_example[0], generate_response))
        entity_replaced_response = self.masked_guess_agent.get_response(entity_replaced_prompt)
        current_joint_memory['generate_guess'] = self.masked_guess_agent.short_memory
        print("Generate Guess Response: {}".format(entity_replaced_response))

        # time.sleep(1)

        # Summarize 
        sentences = (word_with_example[0], word_with_example[1], plain_guess_response, masked_guess_response, entity_replaced_response)
        summarize_prompt = self.summarize_task.get_prompt(sentences)
        summarize_response = self.summarize_agent.get_response(summarize_prompt)
        current_joint_memory['summarize'] = self.summarize_agent.short_memory
        print("Summarize Response: {}".format(summarize_response))
        # time.sleep(1)
        
        return summarize_response, current_joint_memory


# Defining specific parsing functions
def cot_parse_function(response):
    return response.split("结论:")[1].strip()

def icl_parse_function(response):
    return response.split("Meaning:")[-1].strip()

def generate_new_sentences_parse_function(response):
    return response.split("重构后的句子:")[-1].strip()

def masked_guess_parse_function(response):
    return response.split("综合分析:")[-1].strip()

def causal_propose_parse_function(response):

    # Extracts the content for both entity replacement list and reconstructed example
    results = {}

    # Extract entity replacement list
    match_entity_list = re.search(r'\{([^}]*)\}', response, re.DOTALL)
    if match_entity_list:
        results['entity_replacement_list'] = match_entity_list.group(1).strip()
    else:
        results['entity_replacement_list'] = "No entity replacement list found."

    # Extract reconstructed example
    match_reconstructed_example = re.search(r'Reconstructed Sentence:\s*(.*?)\s*(?:\n|$)', response, re.DOTALL)
    if match_reconstructed_example:
        results['reconstructed_example'] = match_reconstructed_example.group(1).strip()
    else:
        results['reconstructed_example'] = "No reconstructed example found."

    return results
# Example usage:
# response = "Your response text with {Entity Replacement List content}"
# parsed_content = parse_entity_replacement_list(response)


def summarize_parse_function(response):
    if "\nMeaning: " in response:
        return response.split("\nMeaning: ")[-1].strip()
    elif "\nDefinition: " in response:
        return response.split("\nDefinition: ")[-1].strip()
    else:
        return response


def default_parser(message):
    return message

# Creating instances of Parser with specific parsing functions
cot_parser = Parser(cot_parse_function)
icl_parser = Parser(icl_parse_function)
masked_guess_parser = Parser(masked_guess_parse_function)
generate_new_sentences_parser = Parser(generate_new_sentences_parse_function)
summarize_parser = Parser(summarize_parse_function)
causal_propose_parser = Parser(causal_propose_parse_function)

def run(model_name='gpt-3.5-turbo', sample_num=3, template_name='Direct'):
    data = get_data()

    use_focus = template_name == 'FOCUS'
    use_causal = template_name == 'Causal'


    all_results = []

    # File naming based on model name and sample number
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    parser = Parser(default_parser)
    if template_name == 'ICL':
        parser = icl_parser
        print("Using ICL parser")
    elif template_name == 'CoT':
        parser = cot_parser
        print("Using CoT parser")


    model_name_list = ['Qwen2-72B-Instruct', 'Qwen2-7B-Instruct']
    sentence_method_name_list = ['SENTENCE_METHOD_ALL']


    # run
    for model_name in model_name_list:
        for sentence_method_name in sentence_method_name_list:

            # load model
            definition_system_prompt = "你是一个资深的词典编撰专家，擅长分析给定词语在若干案例中的含义，并为该词下定义。"

            if model_name in ['GEMINI_15_FLASH_NAME', 'GEMINI_15_PRO_NAME']:
                backbone = GeminiAgent(model_name=model_name, system_prompt=definition_system_prompt,
                                       time_sleep_max=60, time_sleep_min=30)
            elif model_name in ['Qwen2-72B-Instruct', 'Qwen2-7B-Instruct']:
                backbone = QwenAgent(model_name=model_name, system_prompt=definition_system_prompt)
            elif model_name in ['gpt-4o-mini', 'gpt-4o']:
                backbone = GPTAgent(model_name=model_name, system_prompt=definition_system_prompt,
                                    time_sleep_max=20, time_sleep_min=5)
            else:
                raise NotImplementedError
            backbone_name = backbone.model_name.replace(" ", "").replace("/", "_").replace("\\", "_")

            llm = backbone

            templates = URBAN_DICTIONARY_TEMPLATES

            algorithm = FOCUSAlgorithm(llm)


            mean_accumulate_scores = {
                'exact_match': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'bleu_score': 0.0,
                'simcse_score': 0.0
            }

            all_results = []

            for word, sentences in data.items():
                with open('outputs/checked.json', 'r', encoding='utf-8') as f_checked:
                    checked = json.load(f_checked)
                if word in checked:
                    continue
                else:
                    checked.append(word)

                sentences = sentences['examples']
                if sentence_method_name == 'SENTENCE_METHOD_RANDOM' and len(sentences) > sample_num:
                    sample_index = [i for i in range(sample_num)]
                    random.shuffle(sample_index)

                else:
                    sample_index = range(len(sentences))
                word_with_example = (word, [sentences[i] for i in sample_index])#, data_list[i]['raw_example'])
                print(word_with_example)

                enable_bleu_retry = False
                best_prediction = None
                best_score = 0
                retry_count = 0
                max_retries = 5

                prediction, current_joint_memory = algorithm.execute(word_with_example)

                result = {
                    'word': word,
                    # 'prompt': task.get_prompt(word_with_example),
                    'prediction': prediction,
                    'memory': current_joint_memory,
                }
                # current_scores = calculate_metrics(prediction, ref_meaning)
                # for k in mean_accumulate_scores.keys():
                #     mean_accumulate_scores[k] += current_scores[k]
                # print("Current scores: {}".format(current_scores))
                # print("Mean scores: {}".format({k: v/(i+1) for k, v in mean_accumulate_scores.items()}))


                print(f'Current cost: ${LargeLanguageModel.calculate_total_cost():.6f}')

                all_results.append(result)

                output_filename = f"outputs/Urban_{model_name}_{sentence_method_name}_{word}_{template_name}_{sample_num}.json"
                print(f"Output file: {output_filename}")

                # Write to the same output file
                with open(output_filename, "w", encoding="utf-8") as outfile:
                    json.dump(result, outfile, ensure_ascii=False, indent=4)

                with open('outputs/checked.json', 'w', encoding='utf-8') as f_checked:
                    json.dump(checked, f_checked, ensure_ascii=False, indent=4)


                # Write the aggregated results to a JSON file
                new_data = {'output': all_results}
                with open(f"outputs/{model_name}_{sentence_method_name}_aggregated_output.json", "w", encoding="utf-8") as outfile:
                    json.dump(new_data, outfile, ensure_ascii=False ,indent=4)

        break

# Usage
if __name__ == "__main__":
    run(
        model_name='gpt-4o-mini',
        sample_num=10,
        template_name='FOCUS')
    print(f'Total cost: ${LargeLanguageModel.calculate_total_cost():.6f}')
  