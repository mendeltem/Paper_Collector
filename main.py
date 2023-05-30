#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:48:23 2023

@author: temuuleu
"""
from libarary.library import *
from config.config import Config
from sentence_transformers.util import cos_sim
import numpy as np
import time
import requests
    
import PyPDF2
import slate3k
import fitz
from annoy import AnnoyIndex

import seaborn as sns
import os


def download_papers_request(all_paper_result):

    paper_information_df = pd.DataFrame()

    for ri, row in enumerate(all_paper_result.values):
        
        row_df = pd.DataFrame(index=(ri,))
        
        title, url = row[0], row[1]
        
        pdf_url = url+'.pdf'
        response = requests.get(pdf_url)
        
        with open('paper.pdf', 'wb') as file:
            file.write(response.content)
        
        
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
        
        paper_information_df = pd.concat([paper_information_df, row_df])
        
    return paper_information_df



def download_papers_arxiv(all_paper_result):

    paper_information_df = pd.DataFrame()

    for ri, row in enumerate(all_paper_result.values):
        
        row_df = pd.DataFrame(index=(ri,))
        
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
        
        paper_information_df = pd.concat([paper_information_df, row_df])
        
    return paper_information_df


    
def download_pdf_from_link(url,pdf_temp_dir):
    
    try:
        
        paper_url    = url+".pdf"
        paper_name =  paper_url.split("/")[-1]
    
        pdf_url = url.replace('/abs/', '/pdf/') + '.pdf'
        
        paper_path = os.path.join(pdf_temp_dir,paper_name)
    
        # Download the paper
        response = requests.get(pdf_url)
    
        # Save the downloaded content as a PDF file
        with open(paper_path, 'wb') as f:
            f.write(response.content)
            
            
        #check if downloaded pdf is not broken
        # Check if downloaded PDF is not broken
        with open(paper_path, 'rb') as f:
             reader = PyPDF2.PdfReader(f)
             num_pages = len(reader.pages)  # This will fail if the PDF is broken

        
        return paper_path
   
    except Exception as e:
        print(f"Error downloading or verifying PDF: {e}")
        return 1
            
        
    
    
def create_directories():
    
    cfg = Config()
    print(cfg.output_dir)
    """Create necessary directories based on the output directory from the config."""
    output_dir = cfg.output_dir
    directories = [
        "temp_Papers",
        "temp_Papers_pdf",
        "Output_Papers",
        "database"
    ]
    for directory in directories:
        os.makedirs(os.path.join(output_dir, directory), exist_ok=True)
    return output_dir    
    


def process_papers(temp_dir, output_dir, single_word_key_list):
    """Process the papers by concatenating and removing duplicates."""
    search_phrases = create_search_phrases(single_word_key_list)
    search_and_save(single_word_key_list, temp_dir,1)
    file_list  = get_files_from_dir(temp_dir)
    all_paper_result = concat_excel(file_list)
    all_paper_result.drop_duplicates(inplace=True)
    papers_list_path = f"{os.path.join(output_dir, 'Output_Papers')}/papers_list_2.xlsx"
    all_paper_result.to_excel(papers_list_path)
    all_paper_result = pd.read_excel(papers_list_path, index_col=0)
    return all_paper_result 



def extract_pdf_text(paper_path):
    """Extract text from a pdf file using different methods."""
    # Method 1: Using fitz
    doc = fitz.open(paper_path)
    page = doc.loadPage(0)
    text = page.getText("text")

    # Method 2: Using slate3k
    with open(paper_path, 'rb') as f:
        pdf_text = slate3k.PDF(f).text() 

    # Method 3: Using pdfplumber
    with pdfplumber.open(paper_path) as pdf:
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

    # Method 4: Using PyPDF2
    pdf_reader = PyPDF2.PdfReader(paper_path)
    for page_num in range(len(pdf_reader.pages)):
        pdf_text += pdf_reader.pages[page_num].extract_text()
    return pdf_text




def get_embeddings(all_paper_result):
    """Transform data and prepare embeddings using SentenceTransformer."""
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    pd_created_with_embeddings = transform_data(all_paper_result, model)
    data = parallel_preparation(pd_created_with_embeddings)
    ids, embeddings, summarys = data[0], data[1], data[2]
    id_embedding_map = dict(zip(ids, embeddings))
    summary_id_map = dict(zip(summarys, ids))
    return ids, embeddings, summary_id_map, id_embedding_map
    


def compute_similarity_matrix(embeddings):
    """Compute similarity matrix for embeddings."""
    sim = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        start_time = time.time()
        sim[i:,i] = np.round(cos_sim(embeddings[i],embeddings[i:]),2)
        end_time = time.time()
        time_per_iteration = (end_time - start_time) / len(embeddings)
        print(f"Time per iteration: {time_per_iteration} seconds")
    return sim

def main():
    
    cfg = Config()
    print(cfg.output_dir)
        
    output_dir = cfg.output_dir
    temp_dir  = os.path.join(output_dir, "temp_Papers")
    os.makedirs(temp_dir, exist_ok=True)
    pdf_temp_dir  = os.path.join(output_dir, "temp_Papers_pdf")
    os.makedirs(pdf_temp_dir, exist_ok=True)

    categorie_output_dir   = os.path.join(output_dir, "Output_Papers")
    os.makedirs(categorie_output_dir, exist_ok=True)
    
    
    db_dir   = os.path.join(output_dir, "database")
    os.makedirs(db_dir, exist_ok=True)
    
    
    
    papers_list_path   = f"{categorie_output_dir}/papers_list_2.xlsx"


    # single_word_key_list = ['Zero-shot', 'transformer', 'medical', 'learning', 'mri', 'registration',
    #                     'segmentation', 'Brain', 'Neuroimaging', 'lesion-network', 'mapping', 'stroke',
    #                     'lesion', 'infarkt', 'white-matter']
    

    single_word_key_list = ['rna', 'biology',"Candidate", "genes", "positive" ,"selection", "brain"]
    
    search_phrases = create_search_phrases(single_word_key_list)   
    
    search_and_save(single_word_key_list, temp_dir,1)
    
    
    file_list  = get_files_from_dir(temp_dir)
    
    all_paper_result    = concat_excel(file_list)

    all_paper_result.drop_duplicates(inplace=True)
    
    
    all_paper_result.to_excel(papers_list_path)
    
    
    all_paper_result = pd.read_excel(papers_list_path, index_col=0)
    




    model = SentenceTransformer('bert-base-nli-mean-tokens')

    pd_created_with_embeddings = transform_data(all_paper_result, model)
    
    data = parallel_preparation(pd_created_with_embeddings)
    
    ids         = data[0]
    embeddings  = data[1]
    summarys    = data[2]
    
    # Create a dictionary for embeddings and their corresponding IDs
    id_embedding_map = dict(zip(ids, embeddings))
    
    # Create a dictionary for summarys and their corresponding IDs
    summary_id_map = dict(zip(summarys, ids))


    
    # Example: Get the ID for a summary
    example_summary = summarys[0]  # Replace this with the summary you want to use
    example_id_from_summary = summary_id_map.get(example_summary)
    print(f"ID for summary '{example_summary}': {example_id_from_summary}")


    # Example: Get the embeddings for an ID
    example_id = example_id_from_summary
    v = id_embedding_map.get(example_id)
    print(f"Embeddings for ID {example_id}: {example_embeddings}")

    
    sim = np.zeros((len(embeddings), len(embeddings)))
    

    for i in range(len(embeddings)):
        start_time = time.time()
        sim[i:,i] = np.round(cos_sim(embeddings[i],embeddings[i:]),2)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate time per iteration
        time_per_iteration = total_time / len(ids)
        
        print(f"Time per iteration: {time_per_iteration} seconds")
        
    

    #sns.heatmap(sim[:10,:10],annot=True)    
    
    db_file  = os.path.join(db_dir,"db_2.ann")
    
    
    # Assuming 'embeddings' is a list of n-dimensional vectors
    n = len(embeddings[0])
    index = AnnoyIndex(n)  
    
    for i, vec in enumerate(embeddings):
        index.add_item(i, vec)
        
    
    num_trees = 60
    
    index.build(num_trees)  # You can choose the number of trees to build (a higher value gives more accurate results but takes longer)
    
    print(os.path.isdir(os.path.dirname(db_file)))
    
        
    if os.path.exists(db_file):
        os.remove(db_file)
    
    index.save(db_file)

    
    loaded_index = AnnoyIndex(n)
    loaded_index.load(db_file)
    
    # Given an n-dimensional vector 'v'
    K = 5  # Choose the number of nearest neighbors you want to retrieve
    similar_items = loaded_index.get_nns_by_vector(v, K, search_k=-1, include_distances=True)
    
    
    
    print(similar_items)
    
    
    
@timer    
def create_and_save_annoy_index(embeddings, db_dir,  num_trees = 60):
    """Create AnnoyIndex and save to the database directory."""
    n = len(embeddings[0])
    index = AnnoyIndex(n)
    for i, vec in enumerate(embeddings):
        index.add_item(i, vec)

    index.build(num_trees)
    db_file = os.path.join(db_dir,"db_1.ann")
    if os.path.exists(db_file):
        os.remove(db_file)
    index.save(db_file)
    return db_file, n
    
    
def load_and_query_annoy_index(db_file, n, example_embeddings, K=5):
    """Load AnnoyIndex from the database directory and query for similar items."""
    loaded_index = AnnoyIndex(n)
    loaded_index.load(db_file)
    similar_items = loaded_index.get_nns_by_vector(example_embeddings, K, search_k=-1, include_distances=True)
    return similar_items 



if __name__ == "__main__":


    output_dir = create_directories()
    db_dir = os.path.join(output_dir, "database")
    
    
    single_word_key_list = ['rna', 'biology', "Candidate", "genes", "positive" ,"selection", "brain"]
    
    
    all_paper_result = process_papers(os.path.join(output_dir, "temp_Papers"), output_dir, single_word_key_list)
    

    ids, embeddings, summary_id_map, id_embedding_map = get_embeddings(all_paper_result)
    
    
    sim = compute_similarity_matrix(embeddings)
    
    db_file, n = create_and_save_annoy_index(embeddings, db_dir)
    
    # Create a dictionary for embeddings and their corresponding IDs
    id_embedding_map = dict(zip(ids, embeddings))
    
    # Create a dictionary for summarys and their corresponding IDs
    summary_id_map = dict(zip(summarys, ids))
    
    
    # Example: Get the ID for a summary
    example_summary = summarys[0]  # Replace this with the summary you want to use
    example_id_from_summary = summary_id_map.get(example_summary)
    print(f"ID for summary '{example_summary}': {example_id_from_summary}")


    # Example: Get the embeddings for an ID
    example_id = example_id_from_summary
    v = id_embedding_map.get(example_id)
    #print(f"Embeddings for ID {example_id}: {example_embeddings}")
    
    
    example_embeddings = id_embedding_map.get(example_id)
    print(f"ID for summary '{ids[0]}': {example_id}")
    similar_items = load_and_query_annoy_index(db_file, n, example_embeddings)
    print(similar_items)
    
    
    similar_summaries = [summarys[i] for i in similar_items[0]] 
    
    distances_list = [i for i in similar_items[1]] 
    
    for i, summary in enumerate(similar_summaries):
        
        print(summary)
        print(distances_list[i])
        print()
        print()
        print()
    
    
    
    # 'similar_items' contains two lists: the indices of the top K similar vectors and their respective distances


    
    #connections.connect(host='127.0.0.1', port='19530')

    # collection_name = "paper"
    # name ="vector_data"
    # dim = 768

    # paper_id = FieldSchema(
    #   name="id", 
    #   dtype=DataType.INT64, 
    #   is_primary=True, 
    # )

    # paper_summary_vector = FieldSchema(
    #   name=name, 
    #   dtype=DataType.FLOAT_VECTOR, 
    #   dim=dim
    # )

    # schema = CollectionSchema(
    #   fields=[paper_id, paper_summary_vector], 
    #   description="Paper DB"
    # )

    # collection = Collection(
    #     name=collection_name, 
    #     schema=schema, 
    #     using='default', 
    #     shards_num=2,
    #     consistency_level="Strong"
    #     )

    # index_params = {
    #   "metric_type":"L2",
    #   "index_type":"IVF_FLAT",
    #   "params":{"nlist":1024}
    # }

    # collection.create_index(
    #   field_name=name, 
    #   index_params=index_params,
    #   _async=True
    # )
    # print(collection.schema)
    
    # temp_data = all_paper_result.loc[:,:]
    # pd_created_with_embeddings = transform_data(temp_data, model)

    # data = parallel_preparation(pd_created_with_embeddings)
    # mr = collection.insert(data)
    
    """
    
    TODO_SEARCH THE DATABASE
    results = collection.search(
        data=[[,]], 
        anns_field=collection_name, 
        param=index_params, 
        limit=10, 
        expr=None,
        consistency_level="Strong"
    )
    """
    """
    for ri, row in enumerate(all_paper_result.values):
        
        row_df = pd.DataFrame(index=(ri,))
        
        title, url = row[0], row[1]
        
        paper_path   = download_pdf_from_link(url,pdf_temp_dir)
        
        os.path.isfile(paper_path)
        
        doc = fitz.open(paper_path)
        
        page = doc.loadPage(0)  # number of page
        text = page.getText("text")
                
                
        
        
        with open(paper_path, 'rb') as f:
            pdf_text = slate3k.PDF(f).text() 
            
        with pdfplumber.open(paper_path) as pdf:
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text()
    
    
        
        pdf_file = open(paper_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(paper_path)
        pdf_text = ''
        for page_num in range(len(pdf_reader.pages)):
            pdf_text += pdf_reader.pages[page_num].extract_text()
        
    
    #connections.disconnect(alias="default")
    
    
    
    """
    
