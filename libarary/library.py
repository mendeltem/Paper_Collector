#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:49:58 2023

@author: temuuleu
"""

import os
import subprocess
import concurrent.futures
import pandas as pd
import time
import numpy as np
from sentence_transformers.util import cos_sim
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import friedmanchisquare, wilcoxon
import requests
import itertools
from itertools import combinations
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import arxiv

import os
from itertools import combinations
import pandas as pd
from resp.apis.arxiv_api import Arxiv

def timer(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        stop_time = time.time()
        print(f"{f.__name__} : execution time: {stop_time - start_time}")
        return result
    return wrapper


def read_excel(file):
    return pd.read_excel(file, index_col=0)


@timer
def get_subdirectories(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


@timer
def get_files_from_dir(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isfile(os.path.join(path, d))]


def concat_excel(df_result_volumes_file_list):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        dfs = list(executor.map(read_excel, df_result_volumes_file_list))
    return pd.concat(dfs, ignore_index=True)


def format_title(title: str) -> str:
    formatted_title = title.strip().replace(" ", "_").replace(":", "_").replace("-", "_").replace("/", "_")
    return f"{formatted_title}.pdf"


def extract_paper_id(url: str) -> str:
    return url.split("/")[-1]


def fetch_paper_data(paper_id: str) -> arxiv.Result:
    return next(arxiv.Search(id_list=[paper_id]).results())


def encode_summary(model, summary: str):
    return model.encode(summary)


@timer
def store_paper_data(paper_id: int, paper: arxiv.Result, model) -> dict:
    summary_embeddings = encode_summary(model, paper.summary)
    return {
        "id": paper_id,
        "summary_embeddings": summary_embeddings,
    }


@timer
def transform_data(all_paper_result, model):
    pd_created_with_embeddings = {}

    for ri, row in enumerate(all_paper_result.values):
        title, url = row[0], row[1]
        paper_id = extract_paper_id(url)
        paper = fetch_paper_data(paper_id)
        pd_created_with_embeddings[ri] = store_paper_data(ri+1, paper, model)

    return pd_created_with_embeddings


def prepare_data(item):
    key, value = item
    vector = value["summary_embeddings"]
    return key, vector


@timer
def parallel_preparation(pd_created_with_embeddings):
    ids_list = []
    v_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for key, vector in executor.map(prepare_data, pd_created_with_embeddings.items()):
            ids_list.append(key)
            v_list.append(vector)
    
    return [ids_list, v_list]






def create_search_phrases(single_word_key_list):
    combinations_list = []
    for r in range(1, len(single_word_key_list) + 1):
        combinations_list.extend(combinations(single_word_key_list, r))
    phrase_list = [' '.join(phrase) for phrase in combinations_list]
    return phrase_list


def search_and_save(search_phrases, output_dir, max_pages=100, max_dev_temp=1000, ci= 0):
    all_result = pd.DataFrame()

    for ci, phrase in enumerate(search_phrases):
        
        if ci:
            if ci >1:
                break
        
        ap = Arxiv()
        arxiv_result = ap.arxiv(phrase, max_pages=max_pages)
        all_result = pd.concat([all_result, arxiv_result])

        if all_result.shape[0] >= max_dev_temp:
            save_to_excel(all_result[:max_dev_temp], output_dir, ci, max_dev_temp)
            all_result = all_result[max_dev_temp:]

    if all_result.shape[0] > 0:
        save_to_excel(all_result, output_dir, len(search_phrases), max_dev_temp)


def save_to_excel(data_frame, output_dir, ci, max_dev_temp):
    data_part = os.path.join(output_dir, f"papers_{ci+1}_{data_frame.shape[0] // max_dev_temp}.xlsx")
    data_frame.to_excel(data_part)



