# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from concurrent.futures import as_completed, ProcessPoolExecutor
import numpy as np
import scipy
import tqdm
import os
import copy
import functools
import torch

from .utils import Tools, FilePathBuilder, CONSTANTS

class SimilarityScore:
    @staticmethod
    def cosine_similarity(embedding_vec1, embedding_vec2):
        return 1 - scipy.spatial.distance.cosine(embedding_vec1, embedding_vec2)
    
    @staticmethod
    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union

class BoWCodeSearchWorker:
    def __init__(self, repo_embedding_lines, query_embedding, sim_scorer, max_top_k, log_message):
        self.repo_embedding_lines = repo_embedding_lines  # list of embeddings
        self.query_embedding = query_embedding  # embeddings
        self.max_top_k = max_top_k
        self.sim_scorer = sim_scorer
        self.log_message = log_message
    
        
    def _find_top_k_context(self, query_embedding):
        top_k_context = []
        for repo_embedding_line in self.repo_embedding_lines:
            repo_line_embedding = np.array(repo_embedding_line['data'][0]['embedding'])
            similarity_score = self.sim_scorer(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=False)[-self.max_top_k:]
        return top_k_context

    def run(self):
        query_lines_with_retrieved_results = []
        embedding = copy.deepcopy(self.query_embedding)
        top_k_context = self._find_top_k_context(embedding)
        query_lines_with_retrieved_results.append(top_k_context)
        return query_lines_with_retrieved_results

class CodeSearchWrapper:
    def __init__(self, vectorizer, vector_file_path, window_file_path):
        self.vectorizer = vectorizer
        self.max_top_k = 5  # store 20 top k context for the prompt construction (top 10)
        self.vector_file_path = vector_file_path
        self.window_file_path = window_file_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    def _run_bow(self, query_vector):
        workers = []
        # output_path = FilePathBuilder.retrieval_results_path()
        repo_embedding_lines = Tools.load_pickle(self.vector_file_path)
        log_message = f'vector path: {self.vector_file_path} {self.vectorizer}, max_top_k: {self.max_top_k}'
        worker = BoWCodeSearchWorker(repo_embedding_lines, query_vector, SimilarityScore.jaccard_similarity, self.max_top_k, log_message)
        workers.append(worker)
        
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(worker.run, ) for worker in workers}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)
        return results
        
    def _run_dense(self, query_vector):
        repo_embedding_matrix = torch.load(self.vector_file_path)
        if not isinstance(repo_embedding_matrix, torch.Tensor) or repo_embedding_matrix.numel() == 0:
            return []
        repo_embedding_matrix = repo_embedding_matrix.to(self.device)
        window_lines = Tools.load_pickle(self.window_file_path)
        query_vector = query_vector.to(self.device)
        scores = (query_vector @ repo_embedding_matrix.T) * 100
        top_args = torch.argsort(scores, dim=1)[0][-self.max_top_k:]
        try:
            top_context = [{'context': window_lines[i]['context'], 'fpath_tuple': window_lines[i]['metadata'][0]['fpath_tuple']} for i in top_args if i < len(window_lines)]
        except Exception as e:
            print(f'error processing: {self.vector_file_path}')
            print(f'top_args: {len(top_args)}')
            print(f'len of window_lines: {window_lines}')
            raise e
        return top_context
    
    def search_code_context(self, query_vector):
        extension = os.path.splitext(self.vector_file_path)[1]
        
        if extension == '.pkl':
            return self._run_bow(query_vector)
        elif extension == '.pth':
            return self._run_dense(query_vector)
        
        return self._run(query_vector)
        
