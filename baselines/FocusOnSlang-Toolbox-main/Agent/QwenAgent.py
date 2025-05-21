from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from Agent.Agent import Agent
import json_repair
from openai import OpenAI
import traceback
from Constant import *
import os


class QwenAgent(Agent):
    def __init__(self, model_name, system_prompt, task="", device="cuda", max_new_tokens=512, temperature=0.7, seed=10086, is_api=True):
        Agent.__init__(self, model_name, temperature, seed)
        assert model_name in ['Qwen2-7B-Instruct', 'Qwen2-72B-Instruct'], "QwenAgent do not accept " + model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.temperature = temperature
        self.is_api = is_api
        if not self.is_api:
            print("try to load model at " + MODEL_CACHE_FILE + model_name)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_CACHE_FILE + model_name, torch_dtype="auto", device_map="auto",
                                                              local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE_FILE + model_name)
            pass
        else:
            print("init model interface for " + model_name)
            self.model = OpenAI(api_key=MY_API_KEY, base_url=LOCAL_MODEL_SERVICE_FOR_406, timeout=30000)
        self.system_prompt = system_prompt # or task description
        self.seed = seed
        self.task = task
        set_seed(self.seed)

    def query(self, prompt, plain=True, print_prompt=True):
        prompt = prompt[:32768] # in case, prompt is too long
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        if print_prompt:
            print(prompt)
        if not self.is_api:
            with torch.no_grad():
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens,
                                                    temperature=self.temperature)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            resp = self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                seed=self.seed,
                # max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                # logprobs=True,
                # top_logprobs=5
            )
            response = resp.choices[0].message.content
        response = response.lstrip("```json").rstrip("```json")
        res = None
        try:
            if plain:
                res = response
            else:
                res = json_repair.loads(response)
        except Exception as e:
            traceback.print_exc()
            print(response)
            exit(-1)
        return res

    def get_response(self, prompt, plain=True):
        return self.query(prompt, plain)