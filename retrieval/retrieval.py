import torch
from contextlib import contextmanager
from .make_window import ChunckRepoWindowMaker, FunctionRepoWindowMaker
from .build_vector import BagOfWords, BuildVectorWrapper, SFRE400M, SFRE2B
from .search_code import CodeSearchWrapper
from .utils import FilePathBuilder

class RetrievalParam():
    """A data class to hold retrieval parameters. No changes needed here."""
    def __init__(self, window_size, slice_size, vectorizer, window_type, max_line, task_id):
        self.window_size = window_size
        self.slice_size = slice_size
        self.vector_builder_name = vectorizer # Renamed for clarity
        self.window_type = window_type
        self.max_line = max_line
        self.task_id = task_id

class Retrieval():
    
    def __init__(self, repo_dir, retrieval_param, cache_dir):
        """
        Initializes the Retrieval object without loading any models into memory.
        
        The setup is lightweight, only preparing paths and configurations.
        """
        self.repo_dir = repo_dir
        self.cache_dir = cache_dir
        self.retrieval_param = retrieval_param
        
        # Path setup is lightweight and can be done upfront
        self._setup_paths()
        
        # Window maker doesn't load a model, so it's safe to initialize
        self.window_maker = self._build_window_maker()
        self.build_index()

    def _setup_paths(self):
        """Sets up the necessary file paths based on the vectorizer type."""
        builder_name = self.retrieval_param.vector_builder_name
        if builder_name == 'BoW':
            self.window_file_path = FilePathBuilder.repo_windows_path(self.cache_dir, self.repo_dir, self.retrieval_param.window_size, self.retrieval_param.slice_size)
            self.vector_file_path = FilePathBuilder.one_gram_vector_path(self.window_file_path)
        elif builder_name in ['Dense_SFR400M', 'Dense_SFR2B']:
            self.window_file_path = FilePathBuilder.repo_func_windows_path(self.cache_dir, self.repo_dir, self.retrieval_param.max_line, self.retrieval_param.task_id)
            if builder_name == 'Dense_SFR400M':
                self.vector_file_path = FilePathBuilder.sfr_400M_vector_path(self.window_file_path)
            else:
                self.vector_file_path = FilePathBuilder.sfr_2B_vector_path(self.window_file_path)
    
    @contextmanager
    def _vectorization_context(self):
        """
        A context manager to load, use, and then unload the vectorization model.
        This is the core of the memory optimization.
        """
        vector_builder = None
        try:
            # 1. Acquire Resources: Load the model onto the GPU
            print("Acquiring resources: Loading vectorization model...")
            vector_builder = self._build_vector_builder()
            vector_wrapper = BuildVectorWrapper(vector_builder)
            search_wrapper = CodeSearchWrapper(vector_builder, self.vector_file_path, self.window_file_path)
            
            # 2. Yield control to the 'with' block
            yield vector_wrapper, search_wrapper
            
        finally:
            # 3. Release Resources: Unload the model from GPU
            print("Releasing resources: Unloading model to free GPU memory.")
            if vector_builder:
                print("To release")   
                vector_builder.unload_model()
                del vector_builder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def build_index(self):
        """
        Builds the code windows and then vectorizes them.
        This is a heavy, one-time operation that now uses the context manager
        to manage GPU memory.
        """
        print("Step 1: Building repository windows...")
        self.window_maker.build_windows()
        
        print("Step 2: Vectorizing repository windows...")
        with self._vectorization_context() as (vector_wrapper, _):
            vector_wrapper.vectorize_repo_windows()
        print("Index building complete.")

    def retrieve(self, query_text):
        """
        Retrieves relevant code context for a given query.
        The model is loaded only for the duration of this call.
        """
        with self._vectorization_context() as (vector_wrapper, search_wrapper):
            # The model is now in memory
            query_vector = vector_wrapper.vectorize_query(query_text)
            try:
                search_result = search_wrapper.search_code_context(query_vector)
                # Exiting the 'with' block will automatically free the memory
            except Exception as e:
                print(f'error processing query: {query_text}')
                print(f'the query vector is {query_vector}')
                raise e
            return search_result

    def _build_window_maker(self):
        """Factory method for creating a window maker instance."""
        if self.retrieval_param.window_type == 'Chunck':
            return ChunckRepoWindowMaker(self.repo_dir, self.retrieval_param.window_size, self.retrieval_param.slice_size, self.window_file_path)
        elif self.retrieval_param.window_type == 'Function':
            return FunctionRepoWindowMaker(self.repo_dir, self.retrieval_param.max_line, self.window_file_path)
        raise ValueError("Invalid window type specified")
        
    def _build_vector_builder(self):
        """Factory method for creating a vector builder instance."""
        builder_name = self.retrieval_param.vector_builder_name
        if builder_name == 'BoW':
            return BagOfWords(self.window_file_path, self.vector_file_path)
        elif builder_name == 'Dense_SFR400M':
            return SFRE400M(self.window_file_path, self.vector_file_path)
        elif builder_name == 'Dense_SFR2B':
            return SFRE2B(self.window_file_path, self.vector_file_path)
        raise ValueError("Invalid vector builder specified")