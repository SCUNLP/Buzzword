import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
OS_MASTER_PORT = "10086"
OS_MASTER_ADDR = ""

MODEL_CACHE_FILE = "/data/pretrained_models/"
DATASET_CACHE_FILE = "/data/public_datasets/"

QWEN_AGENT_7B_NAME = "Qwen/Qwen2-7B-Instruct"
QWEN_AGENT_72B_NAME = "Qwen/Qwen2-72B-Instruct"

LLAMA3_AGENT_8B_NAME = "llama3/Meta-Llama-3-8B-Instruct"
LLAMA3_AGENT_70B_NAME = "llama3/Meta-Llama-3-70B-Instruct"

GEMINI_15_FLASH_NAME = "gemini-1.5-flash"
GEMINI_15_PRO_NAME = "gemini-1.5-pro"
GEMINI_KEY = "AIzaSyACtdZmlaHy1sWaOmUR7CbEYRSwqa2IcAE"
# GEMINI_KEY = "AIzaSyCGfFH-OjWTqZKN3hZ39mUKEmt5Ghf1O94"
# GEMINI_KEY = "AIzaSyA9s6NJ8IB33Q-hao9Wsg7UaBzefo1RYp4"
GEMINI_KEY_POOL = [

    "AIzaSyARBhYGXnpFinunJKin4vbPyXS37L1yf1w",
    "AIzaSyAmCfC8lF7EN4TUVBWpqELD8DIwBIaZqqM",
    "AIzaSyD2DOjCutvpEgm-4ApZsT1pVLidQpPS5t0",
    "AIzaSyCGfFH-OjWTqZKN3hZ39mUKEmt5Ghf1O94",
    "AIzaSyBrOdfbTwu_vm4os2pagJ1_uwwMrD1x6IQ",

    "AIzaSyCKGj9iogiP6M7vbOBR0gOVYmoiHoTVziA",
    "AIzaSyACtdZmlaHy1sWaOmUR7CbEYRSwqa2IcAE",
    "AIzaSyCrwMOru-e1Hpa4L_Log5FXr57cg2kzqdo",
]
GEMINI_KEY_INDEX = 0

GPT_CHATGPT_NAME = "gpt-3.5-turbo"
GPT_EMBEDDING_NAME = "text-embedding-ada-002"
GPT_GPT4O_NAME = "gpt-4o"
GPT_GPT4O_MINI_NAME = "gpt-4o-mini"
GPT_KEY = "sk-9FOxk0cN3scmcz0bQ5LOweJWAn9gIQHKfre69BpCsCcGKNvJ"
# GPT_KEY = "sk-pj7rhn9MHaOfbMvEZPRlGYpXX2mmRrMubz8XCnvfPhKUzwQ1"
GPT_URL = "https://xiaoai.plus/v1"

BASELINE_DIRECT = "direct"
BASELINE_COT = "cot"
BASELINE_FOCUS = "focus"
BASELINE_MASS = "mass"
BASELINE_SIMPLEDEFINE = "simpledefine"

SENTENCE_METHOD_ALL = "all"
SENTENCE_METHOD_RANDOM = "random"

LOCAL_MODEL_SERVICE_FOR_406 = "http://localhost:10086/v1"
MY_API_KEY = "hc"

MIN_NUM_EXAMPLE_PER_WORD = 1