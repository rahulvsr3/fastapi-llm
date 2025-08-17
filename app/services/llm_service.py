import os
import shutil
from threading import Lock, Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from app.config import MODEL_CACHE_DIR

MODEL_NAME = "TheBloke/OpenLLaMA-3B-GPTQ"

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Globals
tokenizer = None
model = None
generator = None
downloading = False
download_lock = Lock()
request_queue = []

def get_folder_size_gb(folder):
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
    if free_gb < 2:
        raise Exception(f"Not enough storage available ({free_gb:.2f} GB). LLM call aborted.")

def _download_model():
    """Internal function to load model/tokenizer in background"""
    global tokenizer, model, generator, downloading
    with download_lock:
        if model is None:
            downloading = True
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                low_cpu_mem_usage=True,
                cache_dir=MODEL_CACHE_DIR
            )
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            downloading = False

def start_model_download_background():
    """Start model download in a background thread"""
    thread = Thread(target=_download_model, daemon=True)
    thread.start()

def analyze_sentiment(tweets: list):
    """Process request, wait if model is downloading"""
    check_storage_limit()
    global downloading
    while downloading:
        # Wait for model to finish downloading
        time.sleep(1)
    if model is None or tokenizer is None:
        # Ensure model is loaded
        _download_model()
    prompt = "Analyze the sentiment of these tweets and provide a short summary:\n" + "\n".join(tweets)
    output = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]
    summary = output.replace(prompt, "").strip()
    return summary

def get_model_status():
    model_path = os.path.join(MODEL_CACHE_DIR, MODEL_NAME.replace("/", "_"))
    is_downloaded = os.path.isdir(model_path)
    size_gb = get_folder_size_gb(model_path) if is_downloaded else 0
    total, used, free = shutil.disk_usage(MODEL_CACHE_DIR)
    free_gb = free / (1024 ** 3)
    return {
        "model_downloaded": is_downloaded,
        "model_size_gb": round(size_gb, 2),
        "available_storage_gb": round(free_gb, 2),
        "currently_downloading": downloading,
        "queued_requests": len(request_queue)
    }
