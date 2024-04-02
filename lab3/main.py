from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import streamlit as st
import load_models
import re


if not Path(Path.cwd() / 'model').exists():
    load_models.load_model_en_ru()
    load_models.load_model_ru_en()

tokenizer_en = AutoTokenizer.from_pretrained(Path.cwd() / 'model' / 'en_ru_local')
model_en = AutoModelForSeq2SeqLM.from_pretrained(Path.cwd() / 'model' / 'en_ru_local')
tokenizer_ru = AutoTokenizer.from_pretrained(Path.cwd() / 'model' / 'ru_en_local')
model_ru = AutoModelForSeq2SeqLM.from_pretrained(Path.cwd() / 'model' / 'ru_en_local')

def translate_phrase(phrase):
    regex = "^[a-zA-Zа-яА-ЯёЁ!.? ]+$"
    letter_regex="^[a-zA-Z]|[а-яА-Я]|[ёЁ]$"
    pattern = re.compile(regex)
    letter_pattern=re.compile(letter_regex)
    if(phrase==None):
        return "Error: Некорректная фраза для перевода"
    if(phrase==''):
        return "Error: Некорректная фраза для перевода"
    if(pattern.search(phrase) is None):
        return "Error: Некорректная фраза для перевода"
    if(letter_pattern.search(phrase) is None):
        return "Error: Некорректная фраза для перевода"
    lang=detect(phrase)
    if lang == 'ru':
        return translate_phrase_ru(phrase)
    else:
        return translate_phrase_en(phrase)
def translate_phrase_ru(phrase):
    inputs = tokenizer_ru(phrase, return_tensors="pt")
    output = model_ru.generate(**inputs, max_new_tokens=100)
    out_text = tokenizer_ru.batch_decode(output, skip_special_tokens=True)
    return out_text[0]
    
def translate_phrase_en(phrase):
    inputs = tokenizer_en(phrase, return_tensors="pt")
    output = model_en.generate(**inputs, max_new_tokens=100)
    out_text = tokenizer_en.batch_decode(output, skip_special_tokens=True)
    return out_text[0]
    
