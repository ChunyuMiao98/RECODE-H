import os
from collections import defaultdict
from .utils import Tools, FilePathBuilder
import ast

class ChunckRepoWindowMaker:
    def __init__(self, repo, window_size, slice_size, output_path):
        self.repo = repo
        self.window_size = window_size
        self.slice_size = slice_size
        self.slice_step = 1 if window_size // slice_size == 0 else window_size // slice_size
        self.source_code_files = Tools.iterate_repository(repo)
        self.output_path = output_path
        
    def _buid_windows_for_a_file(self, fpath_tuple, code):
        code_windows = []
        code_lines = code.splitlines()
        delta_size = self.window_size // 2
        for line_no in range(0, len(code_lines), self.slice_step): # line_no starts from 0
            start_line_no = max(0, line_no - delta_size)
            end_line_no = min(len(code_lines), line_no + self.window_size - delta_size)
            window_lines = [i for i in code_lines[start_line_no:end_line_no]]
            if not window_lines:  # all empty lines
                continue
            window_text = '\n'.join(window_lines)
            code_windows.append({
                'context': window_text,
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': line_no,
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'repo': self.repo,
                    'slice_size': self.slice_size,
                }
            })
        return code_windows
    
    def _merge_windows_with_same_context(self, code_windows):
        merged_code_windows = defaultdict(list)
        for code_window in code_windows:
            context = code_window['context']
            metadata = code_window['metadata']
            merged_code_windows[context].append(metadata)
        json_lines = []
        for context, metadata_list in merged_code_windows.items():
            json_lines.append({
                'context': context,
                'metadata': metadata_list
            })
        return json_lines

    def build_windows(self):
        all_code_windows = []
        for fpath_tuple, code in self.source_code_files.items():
            all_code_windows += self._buid_windows_for_a_file(fpath_tuple, code)
        merged_code_windows = self._merge_windows_with_same_context(all_code_windows)
        print(f'build {len(merged_code_windows)} windows for {self.repo} with window size {self.window_size} and slice {self.slice_size}')
        print(self.output_path)
        Tools.dump_pickle(merged_code_windows, self.output_path)



class FunctionRepoWindowMaker:
    
    def __init__(self, repo_dir, max_line_num, output_path):
        self.repo_dir = repo_dir
        self.max_line_num = max_line_num
        self.output_path = output_path
        self.source_code_files = Tools.iterate_repository(repo_dir)
        
    def _buid_windows_for_a_file(self, fpath_tuple, code):
        code_windows = []
        code_lines = code.splitlines()
        function_windows = self._extract_functions(code)
        if len(function_windows) == 0:
            return None
        for function_window in function_windows:
            start_line_no = function_window['start_line_no']
            end_line_no = function_window['end_line_no']
            window_lines = [i for i in code_lines[start_line_no - 1 : end_line_no]]
            if not window_lines:
                continue
            window_text = '\n'.join(window_lines)
            code_windows.append({
                'context': window_text,
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': function_window['line_no'],
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.max_line_num,
                    'repo': self.repo_dir,
                }
            })
        return code_windows
    
    def _extract_functions(self, code: str) -> list[dict]:
        """
        Parses Python code and extracts metadata for each function (including async functions)
        defined in the code.

        This method relies on the `end_lineno` attribute of AST function nodes,
        which is reliably available in Python 3.8 and later versions. For older
        Python versions, `end_lineno` might not be present or accurate for all
        function definition scenarios, and this method's accuracy for `end_line_no`
        would be reduced.

        Args:
            self: The instance of the class.
            code: A string containing the Python code to analyze.

        Returns:
            A list of dictionaries. Each dictionary corresponds to a function
            defined in the input code and contains the following keys:
            - "start_line_no": The line number where the function definition (`def` or `async def`) begins.
            - "end_line_no": The line number where the function's body ends.
            - "line_no": The total number of lines spanned by the function definition,
                         calculated as `end_line_no - start_line_no + 1`.

            Returns an empty list if the code cannot be parsed (e.g., due to a
            SyntaxError) or if no functions are defined.
        """
        metadata_list = []
        try:
            # The 'filename' argument can be useful for error reporting,
            # but defaults to '<unknown>' if not provided.
            tree = ast.parse(code)
        except SyntaxError as e:
            # In a class context, you might want to log this error or handle it differently
            # rather than just printing to stdout.
            print(f"SyntaxError parsing input code: {e}")
            return []


        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                # node.end_lineno gives the line number of the last line of the function's suite.
                # This attribute is reliably available in Python 3.8+.
                end_line = node.end_lineno

                if start_line is not None and end_line is not None:
                    total_lines = end_line - start_line + 1
                    metadata_list.append({
                        "start_line_no": start_line,
                        "end_line_no": end_line,
                        "line_no": total_lines
                    })
                else:
                    # This situation is unlikely for FunctionDef/AsyncFunctionDef nodes
                    # in Python 3.8+ parsed from valid code.
                    func_name = getattr(node, 'name', 'Unnamed function')
                    print(f"Warning: Could not accurately determine line numbers for function '{func_name}'. "
                          f"Start line: {start_line}, End line: {end_line}. "
                          "Ensure you are using Python 3.8+ for best results with 'end_lineno'.")
        
        return metadata_list
    
    
    def _merge_windows_with_same_context(self, code_windows):
        merged_code_windows = defaultdict(list)
        for code_window in code_windows:
            context = code_window['context']
            metadata = code_window['metadata']
            merged_code_windows[context].append(metadata)
        json_lines = []
        for context, metadata_list in merged_code_windows.items():
            json_lines.append({
                'context': context,
                'metadata': metadata_list
            })
        return json_lines
    
    # TODO: ADD function namespace metadata
    def build_windows(self):
        all_code_windows = []
        for fpath_tuple, code in self.source_code_files.items():
            code_windows = self._buid_windows_for_a_file(fpath_tuple, code)
            if code_windows is not None:
                all_code_windows += code_windows
        merged_code_windows = self._merge_windows_with_same_context(all_code_windows)
        print(f'build {len(merged_code_windows)} function windows for {os.path.basename(self.repo_dir)}')
        print(self.output_path)
        Tools.dump_pickle(merged_code_windows, self.output_path)
        