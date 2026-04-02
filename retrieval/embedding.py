import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import os
import gc
from contextlib import contextmanager


class SFREmbeddingCode2B:
    def __init__(self):
        self.query_instruction = 'Given Code or Text, retrieval relevant content'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Code-2B_R', trust_remote_code=True, device_map=self.device)
        self.passage_batch = 5
        self.max_length = 32768
        
        self.model.to(self.device)
        print(f"Initialize the SFR Embedding in {self.device}")
    
    def embed_passages(self, passages):
        passage_num = len(passages)//self.passage_batch + 1
        lens = [len(p) for p in passages]
        sort_arg = np.argsort(lens)[::-1]
        passage_sorted = sorted(passages, key=lambda x: len(x), reverse=True)
        embeddings = []
        for i in range(passage_num):
            print(f"process {i*self.passage_batch} to {min(len(passages), (i+1)*self.passage_batch)}")
            batch_passage = passage_sorted[i*self.passage_batch:min(len(passages), (i+1)*self.passage_batch)]
            batch_embeddings = self.model.encode_corpus(batch_passage, max_length=self.max_length)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        index_sort_map = [sort_arg.tolist().index(i) for i in range(len(passages))]
        return embeddings[index_sort_map]
    
    def encode_query(self, query):
        query_embedding = self.model.encode_queries(query, instruction=self.query_instruction, max_length=self.max_length)
        print(query_embedding.shape)
        return F.normalize(query_embedding, p=2, dim=1)
    
    def unload_model(self):
        if self.model is not None:
            print(f"Unloading model '{self.model_name}' from {self.device}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if self.device == 'cuda':
                # This is the crucial part
                torch.cuda.empty_cache()
                
    def _load_model_if_needed(self):
        """Checks if the model is loaded, and if not, loads it onto the device."""
        if self.model is None:
            print(f"Loading model from {self.model_path} to {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            print("Model loaded successfully.")
        
    
class SFREmbeddingCode400M:
    def __init__(self):
        # --- LAZY LOADING CHANGE ---
        # Do not load the model or tokenizer here.
        self.model_path = '/home/linus/code/research_code_gen/models/SFR-Embedding-Code-400M_R'
        self.model = None
        self.tokenizer = None
        
        self.passage_batch = 1
        self.max_length = 8192
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("SFREmbeddingCode400M object initialized (model not loaded).")
        
    def _load_model_if_needed(self):
        """Checks if the model is loaded, and if not, loads it onto the device."""
        if self.model is None:
            print(f"Loading model from {self.model_path} to {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")

    def embed_passages(self, passages):
        self._load_model_if_needed()
        
        with torch.no_grad():
            original_indices = np.argsort([len(p) for p in passages])
            unsort_indices = np.argsort(original_indices)
            passages_sorted = [passages[i] for i in original_indices]

            all_embeddings = []
            for i in range(0, len(passages_sorted), self.passage_batch):
                batch_passages = passages_sorted[i : i + self.passage_batch]
                if not batch_passages:
                    continue

                batch_dict = self.tokenizer(
                    batch_passages,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**batch_dict)
                batch_embeddings = outputs.last_hidden_state[:, 0]
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(batch_embeddings.cpu())

            if not all_embeddings:
                return torch.tensor([]) # Handle case with no passages
                
            embeddings_sorted = torch.cat(all_embeddings, dim=0)
            final_embeddings = embeddings_sorted[unsort_indices]
            return final_embeddings

    def encode_query(self, query):
        self._load_model_if_needed()
        
        with torch.no_grad():
            batch_dict = self.tokenizer(
                query, 
                max_length=self.max_length, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            # Explicitly move each tensor to the device
            for key, value in batch_dict.items():
                batch_dict[key] = value.to(self.device)
            
            outputs = self.model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            del batch_dict
            return F.normalize(embeddings, p=2, dim=1)

    def unload_model(self):
        if self.model is not None:
            print(f"Unloading model '{self.model_path}' from {self.device}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if self.device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
            print("Model unloaded and GPU memory cleared.")