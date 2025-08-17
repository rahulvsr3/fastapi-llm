import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from app.config import MODEL_CACHE_DIR, CODESPACE_STORAGE_LIMIT_GB

MODEL_NAME = "TheBloke/OpenLLaMA-3B-GPTQ"

# Ensure cache folder exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def get_folder_size_gb(folder):
    """Returns folder size in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)

def check_storage_limit():
    total, used, free = shutil.disk_usage(MODEL_CACHE_DIR)
    free_gb = free / (1024 ** 3)
    if free_gb < 2:  # keep at least 2GB free
        raise Exception(f"Not enough storage available ({free_gb:.2f} GB). LLM call aborted.")

# Load model/tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    low_cpu_mem_usage=True,
    cache_dir=MODEL_CACHE_DIR
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def analyze_sentiment(tweets: list):
    check_storage_limit()
    prompt = "Analyze the sentiment of these tweets and provide a short summary:\n" + "\n".join(tweets)
    output = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]
    summary = output.replace(prompt, "").strip()
    return summary

def get_model_status():
    """Returns model status info"""
    model_path = os.path.join(MODEL_CACHE_DIR, MODEL_NAME.replace("/", "_"))
    is_downloaded = os.path.isdir(model_path)
    size_gb = get_folder_size_gb(model_path) if is_downloaded else 0
    total, used, free = shutil.disk_usage(MODEL_CACHE_DIR)
    free_gb = free / (1024 ** 3)
    return {
        "model_downloaded": is_downloaded,
        "model_size_gb": round(size_gb, 2),
        "available_storage_gb": round(free_gb, 2)
    }
