import os
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

MAX_TOKENS_PER_MONTH = 50000
TOKEN_USAGE_FILE = "/tmp/openai_token_usage.txt"

def read_token_usage():
    try:
        with open(TOKEN_USAGE_FILE, "r") as f:
            used = int(f.read().strip())
        return used
    except FileNotFoundError:
        return 0

def write_token_usage(used):
    with open(TOKEN_USAGE_FILE, "w") as f:
        f.write(str(used))

def can_use_tokens(tokens_needed):
    used = read_token_usage()
    return (used + tokens_needed) <= MAX_TOKENS_PER_MONTH

def update_token_usage(tokens_used):
    used = read_token_usage()
    used += tokens_used
    write_token_usage(used)
