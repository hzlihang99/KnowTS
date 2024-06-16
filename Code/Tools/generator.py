import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
from datetime import date, datetime

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from pymongo.mongo_client import MongoClient

import sys
sys.path.append(os.path.dirname(__file__))
from utils import private_openaichat


class BaseGenerator:
    def __init__(self, model_name, num_workers, search_cache=True, insert_cache=True, local_cache=False):
        """Base class for generator, prepared with cache search operation functions"""
        self.model_name = model_name
        self.num_workers = num_workers
        self.insert_cache = insert_cache
        self.search_cache = search_cache
        if local_cache:
            mongo_client = MongoClient("localhost", 27017)
            print('Use the local mongo!')
        else:
            mongo_client = MongoClient(os.environ.get("MONGODB_URI"))
            print('Use the remote mongo!')
        self.mongodb = mongo_client['chat'][os.path.basename(self.model_name).lower()]

    def _search(self, messages):
        id, messages = messages
        mongodb_records = list(self.mongodb.find({'messages':messages}))
        response = None
        if len(mongodb_records) > 0:
            response = sorted(mongodb_records, key=lambda x: x['time'], reverse=True)[0]['response']
        return id, messages, response

    def search(self, messages_list):
        if self.search_cache:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_list = []
                for messages in messages_list:
                    future = executor.submit(self._search, messages)
                    future_list.append(future)
                
                ready_list, target_list = [], []
                pbar = tqdm(total=len(future_list), desc="Search progress:")
                for future in concurrent.futures.as_completed(future_list):
                    id, messages, response = future.result()
                    if response is None:
                        target_list.append((id, messages))
                    else:
                        ready_list.append((id, response))
                    pbar.update()
        else:
            ready_list, target_list = [], messages_list

        return ready_list, target_list

    def generate(self,):
        raise NotImplementedError()


class LocalGenerator(BaseGenerator):
    def __init__(self, model_name, batch_size=1, num_workers=10, max_memory=None, 
                 search_cache=True, insert_cache=True, local_cache=False):
        super().__init__(model_name, num_workers, search_cache, insert_cache, local_cache)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map='auto', 
            max_memory=max_memory, torch_dtype="auto"
        )

    def generate(self, messages_list):
        ready_list, target_list = self.search(messages_list)
        print('*** Find {} from cache, proceed {} for running. ***'.format(len(ready_list), len(target_list)))

        target_response = []
        for i in tqdm(range(int(np.ceil(len(target_list)/self.batch_size))), desc="Inference progress:"):
            model_inputs = []
            for j in range(i*self.batch_size, min(len(target_list),(i+1)*self.batch_size)):
                id, messages = target_list[j]
                model_inputs.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            model_inputs = self.tokenizer(model_inputs, return_tensors="pt", padding=True).to('cuda')
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=256, pad_token_id=self.tokenizer.eos_token_id,
            )

            batch_response = self.tokenizer.batch_decode(generated_ids[:,model_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            target_response.extend([x.strip() for x in batch_response])

        assert len(target_list) == len(target_response)

        print('*** Infer {} done. ***'.format(len(target_list)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_list = []
            for (id, messages), response in zip(target_list,target_response):
                ready_list.append((id, response))
                if self.insert_cache:
                    future_list.append(executor.submit(self.mongodb.insert_one, {"messages": messages, "response":response, 
                                                                                 "time":datetime.now().replace(microsecond=0)}))
            concurrent.futures.wait(future_list, return_when=concurrent.futures.ALL_COMPLETED)

        return ready_list


class RemoteGenerator(BaseGenerator):
    def __init__(self, model_name, num_workers=10, search_cache=True, insert_cache=True, local_cache=False):
        super().__init__(model_name, num_workers, search_cache, insert_cache, local_cache)
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _call(self, id, messages):
        try:
            response = private_openaichat(messages, self.model_name)
            response = (response['choices'][0]['message']['content']).strip()
        except:
            response = self.openai_client.chat.completions.create(
                model=self.model_name, messages=messages,
            )
            response = response.choices[0].message.content.strip()
        
        if self.insert_cache is not None:
            self.mongodb.insert_one({"messages": messages, "response":response, 
                                     "time":datetime.now().replace(microsecond=0)})
        return id, response

    def generate(self, messages_list):
        ready_list, target_list = self.search(messages_list)
        print('*** Find {} from cache, proceed {} for running. ***'.format(len(ready_list), len(target_list)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor: # Fix with 10 for prevent from being baned due to high request
            future_list = []
            for i in range(len(target_list)):
                id, messages = target_list[i]
                future = executor.submit(self._call, id, messages)
                future_list.append(future)

            pbar = tqdm(total=len(future_list), desc="Inference progress:")
            for future in concurrent.futures.as_completed(future_list):
                id, response = future.result()
                ready_list.append((id, response))
                pbar.update()

        print('*** Infer {} done. ***'.format(len(target_list)))

        return ready_list