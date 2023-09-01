from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
import json
from django.contrib.auth.models import User #####
from django.http import JsonResponse , HttpResponse ####
from django.views.decorators.csrf import csrf_exempt
from bs4 import BeautifulSoup
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification

# from utils import get_text_from_url, get_kws
import requests

from collections import namedtuple
import torch, scipy
import numpy as np


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.
    from: https://github.com/huggingface/transformers/blob/77abd1e79f/templates/adding_a_new_example_script/utils_xxx.py

    """
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def get_kws(text, tokenizer, model, max_len = 512, stride = 300):
    inputs = tokenizer(text)
    src_doc = inputs["input_ids"]

    _DocSpan = namedtuple("DocSpan", ["start", "length"])  # pylint: disable=invali
    doc_spans = []
    start_offset = 0

    while start_offset < len(src_doc):
        length = len(src_doc) - start_offset
        if length > max_len:
            length = max_len
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(src_doc):
            break
        start_offset += min(length, stride)


    # processed_input = {i:{} for i in range(len(doc_spans))}
    processed_input = []
    contexts = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = inputs["input_ids"][doc_span.start:doc_span.start + doc_span.length]
        words = tokenizer.decode([id for id in tokens if id not in tokenizer.all_special_ids])
        input = tokenizer(words, return_tensors='pt', truncation=True)
        processed_input.append(input)
        context = []
        for i in range(input['input_ids'].size(1)):
            pos = doc_span.start + i
            is_max_context = _check_is_max_context(doc_spans, doc_span_index, pos)
            context.append(is_max_context)
        contexts.append(context)

    preds = []
    probs = []
    for i, inp in enumerate(processed_input):
      logits = model(**inp).logits
      prob = torch.softmax(logits, dim=2)
      predictionss = torch.where(prob[:,:,1] > 0.25, 1.0, 0.0)#torch.argmax(logits, dim=2)
      word_ids = inp.word_ids()
      predictions = predictionss[0].tolist()
      for i in range(1, len(predictions)):
          if word_ids[i] == word_ids[i-1]:
              predictions[i] = predictions[i-1]
      prob = prob[0][:,1].tolist()
      predicted_token_class = [model.config.id2label[t] for t in predictions]
      preds.append(np.array(predicted_token_class))#[contexts[i]])
      probs.append(np.array(prob))#[contexts[i]])
    kws = []
    processed_probs = []
    i = 0
    for prob, predicted_token_class in zip(probs, preds):
      while i < len(predicted_token_class):
        if predicted_token_class[i] in ['0','LABEL_0']:
          i += 1
        else:
          kw = [inputs["input_ids"][i]]
          cur_prob = [prob[i]]
          i += 1
          while i < len(predicted_token_class):
            if predicted_token_class[i] in ['1','LABEL_1']:
              kw.append(inputs["input_ids"][i])
              cur_prob.append(prob[i])
              i += 1
            else:
              break
          kws.append(kw)
          processed_probs.append(sum(cur_prob)/len(cur_prob))
    kw_txt = [tokenizer.decode(i) for i in kws]

    sorted_kws = [x for _, x in sorted(zip(processed_probs, kw_txt))][::-1]
    return sorted_kws


import trafilatura
from trafilatura.settings import use_config
import re
def get_text_from_url(url):
    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    downloaded = trafilatura.fetch_url(url)
    txt = trafilatura.extract(downloaded, config=config)
    cleaned_text = re.sub(r'http\S+', '', txt)
    cleaned_text = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?', '', cleaned_text)
    
    return cleaned_text


model = AutoModelForSeq2SeqLM.from_pretrained("nkn002/ctrlsum_bart_base")
tokenizer = AutoTokenizer.from_pretrained("nkn002/ctrlsum_bart_base")

extractor = AutoModelForTokenClassification.from_pretrained('nkn002/bert_extractor')
extractor_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def index(request):
    return HttpResponse("Hello, world. You're at the wiki index.")





"""
Nguyen's code: Loading HuggingFace Interface API
"""

API_URL = "https://api-inference.huggingface.co/models/nkn002/ctrlsum_bart_base"
headers = {"Authorization": "Bearer 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"}

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

import trafilatura
from trafilatura.settings import use_config

def get_text_from_url(url):
    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    downloaded = trafilatura.fetch_url(url)
    txt = trafilatura.extract(downloaded, config=config)
    cleaned_text = re.sub(r'http\S+', '', txt)
    cleaned_text = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?', '', cleaned_text)
    
    return cleaned_text


@csrf_exempt
def get_wiki_summary(request):
    data = json.loads(request.body)
    print(data)
    topic = data['topic']
    content = data['content']
    
    cleaned_text = get_text_from_url(content)
    print(cleaned_text)
    
    if topic == "|":
        all_kws = get_kws(cleaned_text, extractor_tokenizer, extractor)
        topic = " | ".join(all_kws)
    data = tokenizer(topic + " => " + cleaned_text, return_tensors="pt", truncation=True)
    input_ids, attention_mask = data["input_ids"], data["attention_mask"]
    output = tokenizer.batch_decode(model.generate(input_ids, 
                                                   attention_mask=attention_mask, 
                                                   num_beams=5, 
                                                   max_new_tokens=256), 
                                    skip_special_tokens=True)[0]

#     del cleaned_text
#     del content
#     output = query({"inputs": topic + " => " + cleaned_text, 
#                     "parameters": {"max_length": 256, 'truncation': 'only_first'},
#                     "options":{'wait_for_model': True}
#                    }) # Nguyen's code: send the topic as input to the model
       
    output = ' '.join(output.split('\n'))
    data = {
        'summary': output,
        'raw': 'Successful',
    }
    
    print('json-data to be sent: ', data)

    return JsonResponse(data)

