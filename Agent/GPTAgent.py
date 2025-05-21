from openai import OpenAI
from Agent.Agent import Agent
from Constant import *
from transformers import set_seed
import json_repair
import time, random


class GPTAgent(Agent):
    def __init__(self, model_name, system_prompt, task="", device="cuda", max_new_tokens=512,
                 temperature=0.7, seed=10086, is_api=True, time_sleep_min=10, time_sleep_max=30):
        Agent.__init__(self, model_name, temperature, seed)
        assert model_name in [GPT_GPT4O_NAME, GPT_GPT4O_MINI_NAME], "GPTAgent do not accept " + model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.temperature = temperature
        self.is_api = is_api
        self.model = OpenAI(api_key=GPT_KEY, base_url=GPT_URL, timeout=300000)
        self.system_prompt = system_prompt # or task description
        self.seed = seed
        self.task = task
        self.time_sleep_min = time_sleep_min
        self.time_sleep_max = time_sleep_max
        set_seed(self.seed)

    def query(self, prompt, print_prompt=False):
        prompt = prompt[:32768] # in case, prompt is too long
        if print_prompt:
            print(prompt)
        ts = random.sample(range(self.time_sleep_min, self.time_sleep_max), 1)[0]
        print("sleep ", ts)
        time.sleep(ts)

        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            seed=self.seed,
            temperature=self.temperature,
        )

        res = completion.choices[0].message.content
        if print_prompt:
            print(res)

        return json_repair.loads(res.replace("```json", "").replace("```", ""))

