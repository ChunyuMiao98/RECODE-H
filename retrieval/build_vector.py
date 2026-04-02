# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tqdm
import itertools
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from .embedding import SFREmbeddingCode400M, SFREmbeddingCode2B
from .utils import Tools, FilePathBuilder
import torch
import os

class BagOfWords:
    def __init__(self, input_file, output_path):
        self.input_file = input_file
        self.output_path = output_path
        self.embedding_name = 'BoW'

    def build(self):
        print(f'building one gram vector for {self.input_file}')
        futures = dict()
        lines = Tools.load_pickle(self.input_file)
        with ProcessPoolExecutor(max_workers=48) as executor:
            for line in lines:
                futures[executor.submit(Tools.tokenize, line['context'])] = line
        
            new_lines = []
            t = tqdm.tqdm(total=len(futures))
            for future in as_completed(futures):
                line = futures[future]
                tokenized = future.result()
                new_lines.append({
                    'context': line['context'],
                    'metadata': line['metadata'],
                    'data': [{'embedding': tokenized}]
                })
                tqdm.tqdm.update(t)
            output_file_path = FilePathBuilder.one_gram_vector_path(self.input_file)
            Tools.dump_pickle(new_lines, self.output_path)
        
    def build_query_vector(self, query_text):
        return Tools.tokenize(query_text)

class SFRE400M:
    def __init__(self, input_file, output_path):
        self.input_file = input_file
        self.output_path = output_path
        self.model_wrapper = SFREmbeddingCode400M()
        self.embedding_name = 'SFRE400M'
        
    def build(self):
        if os.path.exists(self.output_path):
            print(f'SFRE400M vector already exist in {self.input_file}, skip build the vector')
            return 
        print(f'building SFRE400M vector for {self.input_file}')
        lines = Tools.load_pickle(self.input_file)
        contextes = [line['context'] for line in lines]
        # Potential OOM
        embeddings = self.model_wrapper.embed_passages(contextes)
        torch.save(embeddings, self.output_path)
        
    def unload_model(self):
        self.model_wrapper.unload_model()
        
    def build_query_vector(self, query_text):
        return self.model_wrapper.encode_query(query_text)
    

class SFRE2B:
    def __init__(self, input_file, output_path):
        self.input_file = input_file
        self.output_path = output_path
        self.model_wrapper = SFREmbeddingCode2B()
        self.embedding_name = 'SFRE2B'
        
    
    def build(self):
        if os.path.exists(self.output_path):
            print(f'SFRE2B vector already exist in {self.input_file}, skip build the vector')
            return 
        print(f'building SFRE2B vector for {self.input_file}')
        lines = Tools.load_pickle(self.input_file)
        contextes = [line['context'] for line in lines]
        # Potential OOM
        embeddings = self.model_wrapper.embed_passages(contextes)
        torch.save(embeddings, self.output_path)
        
    def build_query_vector(self, query_text):
        return self.model_wrapper.encode_query(query_text)
    

class BuildVectorWrapper:
    def __init__(self, vector_builder):
        self.vector_builder = vector_builder
        self.builder = self.vector_builder

    def vectorize_repo_windows(self):
        self.builder.build()
        
    def vectorize_query(self, query_text):
        return self.builder.build_query_vector(query_text)
        
