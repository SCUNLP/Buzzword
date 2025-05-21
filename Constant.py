import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
OS_MASTER_PORT = "17040"
OS_MASTER_ADDR = ""

MODEL_CACHE_FILE = "/data/pretrained_models/"
DATASET_CACHE_FILE = "/data/public_datasets/"

QWEN_AGENT_7B_NAME = "Qwen/Qwen2-7B-Instruct"
QWEN_AGENT_72B_NAME = "Qwen/Qwen2-72B-Instruct"

LLAMA3_AGENT_8B_NAME = "llama3/Meta-Llama-3-8B-Instruct"
LLAMA3_AGENT_70B_NAME = "llama3/Meta-Llama-3-70B-Instruct"

GEMINI_15_FLASH_NAME = "gemini-1.5-flash"
GEMINI_15_PRO_NAME = "gemini-1.5-pro"
GEMINI_KEY = ""

GEMINI_KEY_POOL = [

]
GEMINI_KEY_INDEX = 0

GPT_CHATGPT_NAME = "gpt-3.5-turbo"
GPT_EMBEDDING_NAME = "text-embedding-ada-002"
GPT_GPT4O_NAME = "gpt-4o"
GPT_GPT4O_MINI_NAME = "gpt-4o-mini"
GPT_KEY = ""
GPT_URL = ""

BASELINE_DIRECT = "direct"
BASELINE_COT = "cot"
BASELINE_FOCUS = "focus"
BASELINE_MASS = "mass"
BASELINE_SIMPLEDEFINE = "simpledefine"

SENTENCE_METHOD_ALL = "all"
SENTENCE_METHOD_RANDOM = "random"
SENTENCE_METHOD_LLM = "llm"
SENTENCE_METHOD_WAUS = "waus"
SENTENCE_METHOD_GDEX = "gdex"

LOCAL_MODEL_SERVICE_FOR_406 = ""
MY_API_KEY = ""

MIN_NUM_EXAMPLE_PER_WORD = 1