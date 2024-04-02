from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
def load_model_en_ru():
    model_name = "Helsinki-NLP/opus-mt-en-ru"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.save_pretrained(Path.cwd() / 'model' / 'en_ru_local')
    model.save_pretrained(Path.cwd() / 'model' / 'en_ru_local')
def load_model_ru_en():
    model_name = "Helsinki-NLP/opus-mt-ru-en"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.save_pretrained(Path.cwd() / 'model' / 'ru_en_local')
    model.save_pretrained(Path.cwd() / 'model' / 'ru_en_local')