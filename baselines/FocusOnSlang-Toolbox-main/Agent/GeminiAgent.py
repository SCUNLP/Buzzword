"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""

import google.generativeai as genai
from Agent.Agent import Agent
import json_repair
import time, random
import traceback


class GeminiAgent(Agent):
    def __init__(self, model_name, system_prompt, max_output_tokens=8192, temperature=0, seed=10086, time_sleep_max=60, time_sleep_min=30):
        Agent.__init__(self, model_name, temperature, seed)
        assert model_name in ['GEMINI_15_FLASH_NAME', 'GEMINI_15_PRO_NAME'], "GeminiAgent do not accept " + model_name
        self.max_output_tokens = max_output_tokens
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.time_sleep_max = time_sleep_max
        self.time_sleep_min = time_sleep_min
        genai.configure(api_key='GEMINI_KEY', transport="rest")
        # Create the model
        # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
            # "seed": seed  #
        }
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
            # {
            #     "category": "HARM_CATEGORY_DEROGATORY",
            #     "threshold": "BLOCK_NONE",
            # },
            {
                "category": "HARM_CATEGORY_SEXUAL",
                "threshold": "BLOCK_NONE",
            },
            # {
            #     "category": "HARM_CATEGORY_TOXICITY",
            #     "threshold": "BLOCK_NONE",
            # }
        ]
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings = self.safety_settings,
            system_instruction = self.system_prompt
            # See https://ai.google.dev/gemini-api/docs/safety-settings
        )

        # self.chat_session = self.model.start_chat(
        #     history=[
        #     ]
        # )

    def query(self, prompt, print_prompt=False):
        prompt = prompt[:32768] # in case, prompt is too long
        if print_prompt:
            print(prompt)
        ts = random.sample(range(self.time_sleep_min, self.time_sleep_max), 1)[0]
        print("sleep ", ts)
        time.sleep(ts)
        response = self.model.generate_content(prompt)
        return json_repair.loads(response.text)

    def query_pool(self, prompt, pool_keys='GEMINI_KEY_POOL', print_prompt=False):
        prompt = prompt[:32768] # in case, prompt is too long
        if print_prompt:
            print(prompt)
        for k in pool_keys:
            try:
                ts = random.sample(range(self.time_sleep_min, self.time_sleep_max), 1)[0]
                print("sleep ", ts)
                time.sleep(ts)

                genai.configure(api_key=k, transport="rest")
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=self.generation_config,
                    safety_settings = self.safety_settings,
                    system_instruction = self.system_prompt
                )

                response = self.model.generate_content(prompt)
                return json_repair.loads(response.text)
            except:
                print("API key wrong ", k)
                # traceback.print_exc()
                time.sleep(30)
        return None

    def get_response(self, user_word):
        return self._get_response(user_word)

