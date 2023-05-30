#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:49:58 2023

@author: temuuleu
"""

import os
import concurrent.futures
import pandas as pd
import time


from concurrent.futures import ProcessPoolExecutor
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
from itertools import combinations
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import arxiv
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
        "summary":paper.summary
    }


@timer
def create_dictionary(all_paper_result, model):
    dictionary_with_embeddings = {}

    for ri, row in enumerate(all_paper_result.values):
        title, url = row[0], row[1]
        paper_id = extract_paper_id(url)
        paper = fetch_paper_data(paper_id)
        
        summary_embeddings = encode_summary(model, paper.summary)

        dictionary_with_embeddings[ri] = {}
        dictionary_with_embeddings[ri]["title"] = paper.title
        dictionary_with_embeddings[ri]["summary_embeddings"] = summary_embeddings
        dictionary_with_embeddings[ri]["paper"] = paper
        dictionary_with_embeddings[ri]["summary"] = paper.summary
        

    return dictionary_with_embeddings
    

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
    summary = value["summary"]
    
    return key, vector,summary


@timer
def parallel_preparation(pd_created_with_embeddings):
    ids_list = []
    v_list = []
    s_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for key, vector,summary in executor.map(prepare_data, pd_created_with_embeddings.items()):
            ids_list.append(key)
            v_list.append(vector)
            s_list.append(summary)
    
    return [ids_list, v_list,s_list]


def prepare_data_all(item):
    key, value  = item
    vector      = value["summary_embeddings"]
    
    title       = value["title"]
    summary     = value["summary"]
    paper       = value["paper"]
    
    return key, vector,title,summary,paper

def create_search_phrases(single_word_key_list):
    combinations_list = []
    for r in range(1, len(single_word_key_list) + 1):
        combinations_list.extend(combinations(single_word_key_list, r))
    phrase_list = [' '.join(phrase) for phrase in combinations_list]
    return phrase_list


def process_paper(row):
    start_time = time.time()
    row_df = pd.DataFrame()
    title, url = row[0], row[1]
    paper_id = extract_paper_id(url)
    paper    = fetch_paper_data(paper_id)
    
    row_df["title"]  =   paper.title
    row_df["summary"]  =   paper.summary
    row_df["published"]  = paper.published.date()
    row_df["primary_category"]  =   paper.primary_category
    row_df["Author"]  =   ";".join(str(paper.authors[author]) for author in range(len(paper.authors)))  
    row_df["pdf_url"]  =   paper.pdf_url
    row_df["links"]  =   ";".join(str(paper.links[author]) for author in range(len(paper.links)))  
    row_df["paper_id"]  = paper_id
    
    end_time = time.time()
    print(f"Time taken for process_paper: {end_time - start_time} seconds")
    
    return row_df



def search_and_save(search_phrases, output_dir, max_pages=1, max_dev_temp=2, ci= 0):
    all_result = pd.DataFrame()
    
    for ci, phrase in enumerate(search_phrases):
        
        if ci:
            if ci >1:
                break
        
        ap = Arxiv()
        arxiv_result = ap.arxiv(phrase, max_pages=max_pages)
        
        all_result = pd.concat([all_result, arxiv_result])
        
        
        
        
        
        
    # paper_information_df = pd.DataFrame()
    
    # for row in all_result.values:
    #     print(f"row {row}")
        
    #     try:
    #         paper_information_df = pd.concat([paper_information_df, process_paper(row)])
    #     except:
    #         print("failed")
    
    
    #return paper_information_df

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
        
        

# def search_and_save(search_phrases, output_dir, max_pages=100, max_dev_temp=1000, ci= 0):
#     all_result = pd.DataFrame()
    
#     paper_information_df = pd.DataFrame()

#     for ci, phrase in enumerate(search_phrases):
        
#         if ci:
#             if ci >1:
#                 break
        
#         ap = Arxiv()
#         arxiv_result = ap.arxiv(phrase, max_pages=max_pages)
        
#         for ri, row in enumerate(arxiv_result.values):
            
#             print(f"index {ri}")
            
#             row_df = pd.DataFrame(index=(ri,))
#             title, url = row[0], row[1]
#             paper_id = extract_paper_id(url)
#             paper    = fetch_paper_data(paper_id)
            
#             row_df["title"]  =   paper.title
#             row_df["summary"]  =   paper.summary
#             row_df["published"]  = paper.published.date()
#             row_df["primary_category"]  =   paper.primary_category
#             row_df["Author"]  =   ";".join(str(paper.authors[author]) for author in range(len(paper.authors)))  
#             row_df["pdf_url"]  =   paper.pdf_url
#             row_df["links"]  =   ";".join(str(paper.links[author]) for author in range(len(paper.links)))  
#             row_df["paper_id"]  = paper_id
            
#             print(f"title {paper.title}")
#             paper_information_df = pd.concat([paper_information_df, row_df])
            
#     return paper_information_df

        



def save_to_excel(data_frame, output_dir, ci, max_dev_temp):
    
    data_part = os.path.join(output_dir, f"papers_{ci+1}_{data_frame.shape[0] // max_dev_temp}.xlsx")
    print(f"data_part {data_part}")
    data_frame.to_excel(data_part)



