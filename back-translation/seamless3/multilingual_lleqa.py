import os
print("The current working directory is ", os.getcwd())
print("The files in the current working directory are ", os.listdir("."))
print("Files created, last modified etc is ", os.system("stat *"))

import datetime
print("Time now is ", datetime.datetime.now())

import io
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import mmap
import numpy
import soundfile
import torchaudio
import torch

from collections import defaultdict
from pathlib import Path
from pydub import AudioSegment

from seamless_communication.inference import Translator
from seamless_communication.streaming.dataloaders.s2tt import SileroVADSilenceRemover

model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

translator = Translator(
    model_name,
    vocoder_name,
    device=torch.device("cuda"),
    dtype=torch.float16,
)


def translate_sent(sentence, tgt_lang=None, src_lang=None):
  if sentence:
    text_output, speech_output = translator.predict(
        input = sentence[:4096],
        task_str = "t2tt",
        tgt_lang = tgt_lang,
        src_lang = src_lang
    )
    return str(text_output[0])
  else:
    return None

from datasets import load_dataset, Dataset, DatasetDict
#https://huggingface.co/datasets/maastrichtlawtech/lleqa
lleqa_questions = load_dataset("maastrichtlawtech/lleqa", "questions")
lleqa_corpus = load_dataset("maastrichtlawtech/lleqa", "corpus")

import pandas as pd

def translate_lleqa_questions(tgt_lang="eng", src_lang="fra"):
  print("Beginning translation of training set of questions to " + tgt_lang + " from " + src_lang)
  train_rows = []
  for train_row in lleqa_questions['train']:
    temp_row = {}
    temp_row["id"] = train_row["id"]
    temp_row["question"] = translate_sent(train_row["question"], tgt_lang = tgt_lang, src_lang = src_lang)
    temp_row["answer"] = translate_sent(train_row["answer"], tgt_lang = tgt_lang, src_lang = src_lang)
    temp_row["regions"] = []
    for region in train_row["regions"]:
      temp_row["regions"].append(translate_sent(region, tgt_lang = tgt_lang, src_lang = src_lang))
    temp_row["topics"] = []
    for topic in train_row["topics"]:
      temp_row["topics"].append(translate_sent(topic, tgt_lang = tgt_lang, src_lang = src_lang))
    temp_row["article_ids"] = train_row["article_ids"]
    temp_row["paragraph_ids"] = train_row["paragraph_ids"]
    train_rows.append(temp_row)
    print("Number of questions in the training set is ",len(train_rows))

  train_df = pd.DataFrame(train_rows)
  train_hfdataset = Dataset.from_pandas(train_df)

  print("Beginning translation of validation set of questions to " + tgt_lang + " from " + src_lang)
  val_rows = []
  for val_row in lleqa_questions['validation']:
    temp_row = {}
    temp_row["id"] = val_row["id"]
    temp_row["question"] = translate_sent(val_row["question"], tgt_lang = tgt_lang, src_lang = src_lang)
    temp_row["answer"] = translate_sent(val_row["answer"], tgt_lang = tgt_lang, src_lang = src_lang)
    temp_row["regions"] = []
    for region in val_row["regions"]:
      temp_row["regions"].append(translate_sent(region, tgt_lang = tgt_lang, src_lang = src_lang))
    temp_row["topics"] = []
    for topic in val_row["topics"]:
      temp_row["topics"].append(translate_sent(topic, tgt_lang = tgt_lang, src_lang = src_lang))
    temp_row["article_ids"] = val_row["article_ids"]
    temp_row["paragraph_ids"] = val_row["paragraph_ids"]
    val_rows.append(temp_row)
    print("Number of questions in the validation set is ", len(val_rows))

  val_df = pd.DataFrame(val_rows)
  val_hfdataset = Dataset.from_pandas(val_df)

  print("Beginning translation of test set of questions to " + tgt_lang + " from " + src_lang)
  test_rows = []
  for test_row in lleqa_questions['test']:
    temp_row = {}
    temp_row["id"] = test_row["id"]
    temp_row["question"] = translate_sent(test_row["question"], tgt_lang = tgt_lang, src_lang = src_lang)
    temp_row["answer"] = translate_sent(test_row["answer"], tgt_lang = tgt_lang, src_lang = src_lang)
    temp_row["regions"] = []
    for region in test_row["regions"]:
      temp_row["regions"].append(translate_sent(region, tgt_lang = tgt_lang, src_lang = src_lang))
    temp_row["topics"] = []
    for topic in test_row["topics"]:
      temp_row["topics"].append(translate_sent(topic, tgt_lang = tgt_lang, src_lang = src_lang))
    temp_row["article_ids"] = test_row["article_ids"]
    temp_row["paragraph_ids"] = test_row["paragraph_ids"]
    test_rows.append(temp_row)
    print("Number of questions in the test set is ", len(test_rows))

  test_df = pd.DataFrame(test_rows)
  test_hfdataset = Dataset.from_pandas(test_df)

  final_dataset = DatasetDict({"train": train_hfdataset, "validation": val_hfdataset, "test":test_hfdataset})
  return final_dataset

def translate_lleqa_corpus(tgt_lang="eng", src_lang="fra"):
  print("Beginning translation of corpus to ", tgt_lang, " from ", src_lang)
  corpus_rows = []
  for corpus_row in lleqa_corpus["corpus"]:
    temp_corpus_row = {}
    temp_corpus_row["id"] = corpus_row["id"]
    temp_corpus_row["article"] = translate_sent(corpus_row["article"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["description"] = translate_sent(corpus_row["description"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["authority"] = translate_sent(corpus_row["authority"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["reference"] = translate_sent(corpus_row["reference"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["code"] = translate_sent(corpus_row["code"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["article_no"] = corpus_row["article_no"]
    temp_corpus_row["book"] = translate_sent(corpus_row["book"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["part"] = translate_sent(corpus_row["part"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["act"] = translate_sent(corpus_row["act"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["chapter"] = translate_sent(corpus_row["chapter"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["section"] = translate_sent(corpus_row["section"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["subsection"] = translate_sent(corpus_row["subsection"], tgt_lang=tgt_lang, src_lang=src_lang)
    temp_corpus_row["paragraphs"] = {}
    for key, value in corpus_row["paragraphs"].items():
      temp_corpus_row["paragraphs"][key] = translate_sent(corpus_row["paragraphs"][key], tgt_lang=tgt_lang, src_lang=src_lang)
    corpus_rows.append(temp_corpus_row)
    print("Number of rows in the corpus is ", len(corpus_rows))

  corpus_df = pd.DataFrame(corpus_rows)
  corpus_hfdataset = Dataset.from_pandas(corpus_df)
  dataset_dict = DatasetDict({"corpus":corpus_hfdataset})
  return dataset_dict

import pickle

def prepare_dataset(tgt_lang="eng", src_lang="fra"):
    multilingual_lleqa_corpus = translate_lleqa_corpus(tgt_lang=tgt_lang, src_lang=src_lang)
    multilingual_lleqa_questions = translate_lleqa_questions(tgt_lang=tgt_lang, src_lang=src_lang)
    return (multilingual_lleqa_corpus, multilingual_lleqa_questions, tgt_lang, src_lang)

#english_dataset = prepare_dataset(tgt_lang="eng", src_lang="fra")
#print("Saving English Dataset in englishlleqa.pickle")
#with open("englishlleqa.pickle", "wb") as english:
#  pickle.dump(english_dataset, english)

#print()
#print("Files created, last modified etc is ", os.system("stat *"))

#japanese_dataset = prepare_dataset(tgt_lang="jpn", src_lang="fra")
#print("Saving Japanese Dataset in japaneselleqa.pickle")
#with open("japaneselleqa.pickle", "wb") as japanese:
#  pickle.dump(japanese_dataset, japanese)

#print()
#print("Files created, last modified etc is ", os.system("stat *"))

#italian_dataset = prepare_dataset(tgt_lang="ita", src_lang="fra")
#print("Saving Italian Dataset in italianlleqa.pickle")
#with open("italianlleqa.pickle", "wb") as italian:
#  pickle.dump(italian_dataset, italian)

#print()
#print("Files created, last modified etc is ", os.system("stat *"))

finnish_dataset = prepare_dataset(tgt_lang="fin", src_lang="fra")
print("Saving Finnish Dataset in finnishlleqa.pickle")
with open("finnishlleqa.pickle", "wb") as finnish:
  pickle.dump(finnish_dataset, finnish)

print()
print("Files created, last modified etc is ", os.system("stat *"))

dutch_dataset = prepare_dataset(tgt_lang="nld", src_lang="fra")
print("Saving Dutch Dataset in dutchlleqa.pickle")
with open("dutchlleqa.pickle", "wb") as dutch:
  pickle.dump(dutch_dataset, dutch)

print()
print("Files created, last modified etc is ", os.system("stat *"))

spanish_dataset = prepare_dataset(tgt_lang="spa", src_lang="fra")
print("Saving Spanish Dataset in spanishlleqa.pickle")
with open("spanishlleqa.pickle", "wb") as spanish:
  pickle.dump(spanish_dataset, spanish)

print()
print("Files created, last modified etc is ", os.system("stat *"))

multilingual_datasets = {
	"finnish" : finnish_dataset,
	"dutch" : dutch_dataset,
	"spanish" : spanish_dataset
}

with open("multilingualdatasets.pickle", "wb") as multilingdatasets:
    pickle.dump(multilingual_datasets, multilingdatasets)

print()
print("Files created, last modified etc is ", os.system("stat *"))
