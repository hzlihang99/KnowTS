import os
import json
import http.client
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
from datetime import date, datetime
import numpy as np
from sentence_transformers import SentenceTransformer

import multiprocessing as mp
from google.cloud import translate_v2 as translate
translate_client = translate.Client()

from openai import OpenAI
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
from pymongo.mongo_client import MongoClient
mongo_client = MongoClient(os.environ.get("MONGODB_URI"))


def private_openaichat(messages, model_name):
    conn = http.client.HTTPSConnection("api2.aigcbest.top")
    payload = json.dumps({"model": model_name, "messages": messages})
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer {}'.format(os.environ.get("QDD_API_KEY")),
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    response = conn.getresponse()
    response = json.loads(response.read().decode("utf-8"))
    return response


def private_openaiembed(input, model_name):
    conn = http.client.HTTPSConnection("api2.aigcbest.top")    
    payload = json.dumps({"model": model_name, "input": input})
    headers = {
        'Authorization': 'Bearer {}'.format(os.environ.get("QDD_API_KEY")),
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/embeddings", payload, headers)
    response = conn.getresponse()
    response = json.loads(response.read().decode("utf-8"))
    return response


def textlist_openaiembed(text_list, model_name, batch_size=1024):
    input_list = list(set(text_list))
    batch_num = int(np.ceil(len(input_list)/batch_size))
    output_list = []
    for i in tqdm(range(batch_num)):
        batch_begin = i * batch_size
        batch_end = min((i+1) * batch_size, len(input_list))
        response = openai_client.embeddings.create(
            input=input_list[batch_begin:batch_end], model=model_name
        )
        output_list.extend([np.array(x.embedding) for x in response.data])
    output_dict = dict(list(zip(input_list, output_list)))
    return np.array([output_dict[x] for x in text_list])


def textlist_sbertembed(text_list, model_name, batch_size=1024):
    model = SentenceTransformer(model_name)
    input_list = list(set(text_list))
    batch_num = int(np.ceil(len(input_list)/batch_size))
    output_list = []
    for i in tqdm(range(batch_num)):
        batch_begin = i * batch_size
        batch_end = min((i+1) * batch_size, len(input_list))
        output_list.append(model.encode(input_list[batch_begin:batch_end]))
    output_list = np.concatenate(output_list, axis=0)
    output_dict = dict(list(zip(input_list, output_list)))
    return np.array([output_dict[x] for x in text_list])


def run_translate(source_text, target_language = 'en', cache_db = None):
    mongodb_records = []
    if cache_db is not None:
        mongodb_records = list(cache_db.find({'source_text':source_text}))
    if len(mongodb_records) == 0:
        response = translate_client.translate(source_text, target_language)
        response = response['translatedText']
        cache_db.insert_one({"source_text": source_text, 
                             "response":response, "target_language": target_language, 
                             "time":datetime.now().replace(microsecond=0)})
    else:
        response = sorted(mongodb_records, key=lambda x: x['time'], reverse=True)[0]['response']
    return source_text, response


def text2prompt(knowledge, question):
    prompt = '\n'.join(["<Knowledge>: {}".format(knowledge), "<Question>: {}".format(question)])
    return prompt


def prepare_zerohot(request_text, style='cot'):
    if style == 'regular':
        system_text = ["You are a knowledge concept annotator.",
                       "Your job is to judge whether the <Question> is concerning the <Knowledge>.",
                       "Please respond with <Yes> or <No>.",]
    elif style == 'reason':
        system_text = ["You are a knowledge concept annotator.",
                       "Your job is to judge whether the <Question> is concerning the <Knowledge>.",
                       "The judgement token: '<Yes>' or '<No>' should be provided at the start of the response.",
                       "You should also provide the judging reasons for your judgement.",]
    else:
        system_text = ["You are a knowledge concept annotator.",
                       "Your job is to judge whether the <Question> is concerning the <Knowledge>.",
                       "Your should first provide the judging reasons before give your judgement.",
                       "The judgement token: '<Yes>' or '<No>' should be provided at the end of the response."]    
    system_text = ' '.join(system_text)

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": request_text},
    ]
    return messages