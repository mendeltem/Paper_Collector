#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:48:23 2023

@author: temuuleu
"""



from libarary.library import *

from config.config import Config


def main():
    
    cfg = Config()

    print(cfg.output_dir)
        
    
    output_dir = cfg.output_dir
    temp_dir  = os.path.join(output_dir, "temp_Papers")
    os.makedirs(output_dir, exist_ok=True)

    
    categorie_output_dir   = os.path.join(output_dir, "Output_Papers")

    papers_list_path   = f"{categorie_output_dir}/papers_list.xlsx"

    os.makedirs(categorie_input_dir, exist_ok=True)
    os.makedirs(categorie_output_dir, exist_ok=True)
    

    single_word_key_list = ['Zero-shot', 'transformer', 'medical', 'learning', 'mri', 'registration',
                        'segmentation', 'Brain', 'Neuroimaging', 'lesion-network', 'mapping', 'stroke',
                        'lesion', 'infarkt', 'white-matter']

    search_phrases = create_search_phrases(single_word_key_list)   
    search_and_save(search_phrases, temp_dir,1)
    
    file_list  = get_files_from_dir(temp_dir)
    all_paper_result    = concat_excel(file_list)
    all_paper_result.drop_duplicates(inplace=True)
    all_paper_result.to_excel(papers_list_path)
    
    
    

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    connections.connect(host='127.0.0.1', port='19530')

    collection_name = "paper"
    name ="vector_data"
    dim = 768

    paper_id = FieldSchema(
      name="id", 
      dtype=DataType.INT64, 
      is_primary=True, 
    )

    paper_summary_vector = FieldSchema(
      name=name, 
      dtype=DataType.FLOAT_VECTOR, 
      dim=dim
    )

    schema = CollectionSchema(
      fields=[paper_id, paper_summary_vector], 
      description="Paper DB"
    )

    collection = Collection(
        name=collection_name, 
        schema=schema, 
        using='default', 
        shards_num=2,
        consistency_level="Strong"
        )

    index_params = {
      "metric_type":"L2",
      "index_type":"IVF_FLAT",
      "params":{"nlist":1024}
    }

    collection.create_index(
      field_name=name, 
      index_params=index_params,
      _async=True
    )
    print(collection.schema)
    temp_data = all_paper_result.loc[:10,:]
    pd_created_with_embeddings = transform_data(temp_data, model)

    data = parallel_preparation(pd_created_with_embeddings)
    mr = collection.insert(data)
    
    """
    results = collection.search(
        data=[[,]], 
        anns_field=collection_name, 
        param=index_params, 
        limit=10, 
        expr=None,
        consistency_level="Strong"
    )
    """
        
        
        
    
    connections.disconnect(alias="default")
    
    
    
    
    
