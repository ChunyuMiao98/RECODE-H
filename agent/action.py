import random
from copy import copy
from abc import abstractmethod
from utils import *
from retrieval import Retrieval, RetrievalParam
from constants import RESPONSE_FORMAT

class CommandArgs:
    pass

class Command:
    def __init__(self):
        self.cmd_type = "OTHER"

    @abstractmethod
    def docstring(self) -> str:
        pass

    @abstractmethod
    def execute_command(self, *args) -> str:
        pass


"""
@@@@@@@@@@@@@@@@@@
@@ CODING TOOLS @@
@@@@@@@@@@@@@@@@@@
"""

class Replace(Command):
    def __init__(self, args, task_id):
        super().__init__()
        self.cmd_type = "CODE-replace"
        self.task_id = task_id
        self.work_dir = args.work_dir

    def docstring(self) -> str:
        return (
            "============= REWRITE CODE EDITING TOOL =============\n"
            "You also have access to a code replacing tool. \n"
            "This tool allows you to entirely re-write/replace all of the current code and erase all existing code.\n"
            "You can use this tool via the following command: ```REPLACE\n<code here>\n```, where REPLACE is the word REPLACE and <code here> will be the new code that is replacing the entire set of old code. This tool is useful if you want to make very significant changes, such as entirely changing the code file content. Try limiting the use of rewriting and aim for editing the code more."
        )

    def execute_command(self, command) -> str:
        code_content = self.clean_code(command['parameters']["code"])
        DatasetManager.write_code_content(self.work_dir, self.task_id, code_content)
        return True, 'Command replace execution success'
    
    @staticmethod
    def clean_code(code_segment: str) -> str:
        '''
        Clean the code string by removing leading and tailing ```python and ``` characters.
        '''
        # Strip whitespace first
        code_segment = code_segment.strip()
        
        # Remove opening code fence (```python, ```py, or just ```)
        if code_segment.startswith("```python"):
            code_segment = code_segment[len("```python"):]
        elif code_segment.startswith("```py"):
            code_segment = code_segment[len("```py"):]
        elif code_segment.startswith("```"):
            code_segment = code_segment[len("```"):]
        
        
        # Remove closing code fence
        if code_segment.endswith("```"):
            code_segment = code_segment[:-len("```")]
        elif code_segment.endswith("\"\"\""):
            code_segment = code_segment[:-len("\"\"\"")]
        
        # Strip any remaining whitespace
        return code_segment
        



class Edit(Command):
    def __init__(self,args, task_id):
        super().__init__()
        self.cmd_type = "CODE-edit"
        self.task_id = task_id
        self.work_dir = args.work_dir

    def docstring(self) -> str:
        return (
            "============= CODE EDITING TOOL =============\n"
            "You also have access to a code editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with code.\n"
            "You can edit code using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new lines to replace old lines> will be the new code that is replacing the old code.\n"
            "Your output should match the original code as closely as possible(eg. indentation). The correct indentation is very import when using this toll.\n"
        )

    def execute_command(self, command) -> str:
        code_content = DatasetManager.load_code_raw_content(self.work_dir, self.task_id)
        lines = code_content.splitlines()
        code = self.clean_code(command['parameters']["code"])
        new_lines = code.splitlines()
        start = command['parameters']['start']
        end = command['parameters']['end']
        new_lines = lines[:start - 1] + new_lines + lines[end:]
        DatasetManager.write_code_content(self.work_dir, self.task_id, '\n'.join(new_lines))
        return True, 'Command edit execution success'
    
    @staticmethod
    def clean_code(code_segment: str) -> str:
        '''
        Clean the code string by removing leading and tailing ```python and ``` characters.
        '''
        if code_segment.startswith("```python"):
            code_segment = code_segment[len("```python"):]

        if code_segment.endswith("```"):
            code_segment = code_segment[: -len("```")]
        return code_segment


class Read(Command):
    
    def __init__(self, args, task_id):
        super().__init__()
        self.cmd_type = "CODE-read"
        self.task_id = task_id
        self.work_dir = args.work_dir
        
    def docstring(self) -> str:
        return (
                '============= CODE READ TOOL =============\n',
                'You also have access to a code reading tool.\n',
                'This tool allows you to read the content of any file in the current repository. It helps you understand the codebase context before making changes or generating new code.\n',
                'You can read a file using the following command:\n',
                'READ <file path to be read>\n',
                'READ is the word READ, and <file path to be read> is the relative path of the file you want to inspect. This command will return the full content of the specified file.\n',
                'Use this command when you need to examine the contents of any file in the repository, including the current target file.\n'
            )
    def execute_command(self, command) -> str:
        file_path = command['parameters']['file_path']
        code_content = DatasetManager.load_repo_content(self.work_dir, self.task_id, file_path)
        return True, code_content


class Retrieve(Command):
    
    def __init__(self, args, task_id):
        super().__init__()
        self.cmd_type = "CODE-retrieve"
        self.retriever = self.build_retriever(args, task_id)
        self.repo_name = DatasetManager.load_repo_name(args.work_dir, task_id)
    
    def build_retriever(self, args, task_id):
        param = RetrievalParam(window_size=None, slice_size=None, vectorizer='Dense_SFR400M', window_type='Function', max_line=0, task_id=task_id)
        repo_dir = DatasetManager.code_repo_dir(args.work_dir, task_id)
        cache_dir = DatasetManager.load_cache_dir(args.work_dir)
        return Retrieval(repo_dir=repo_dir, cache_dir=cache_dir, retrieval_param=param)
    
    def docstring(self) -> str:
        return (
            '============= CODE RETRIEVE TOOL =============\n',
            'You also have access to a code retrieval tool.\n',
            'This tool allows you to retrieve the most relevant code functions for a given natural language query. It helps you understand where and how specific functionality is implemented in the codebase.\n',
            'You can retrieve code using the following command:\nRETRIEVE\nQUERY\n',
            'RETRIEVE is the word RETRIEVE, and QUERY is your request in natural language. The response will return the top relevant functions, including their code and location in the repository. This will be the primary way to locate and explore code related to specific functionality or concepts.\n',
            'Use this command when you want to investigate or modify code related to a particular feature, action, or behavior.\n'
        )
        
    def execute_command(self, command) -> str:
        try:
            top_k_functions = self.retriever.retrieve(command['parameters']['query'])
        except Exception as e:
            print(f"Error when process command: {command}")
            raise e
        code_content_str = ''
        for function_code in reversed(top_k_functions):
            function_location_tuple = function_code['fpath_tuple']
            if self.repo_name not in function_location_tuple:
                print(f"repo name '{self.repo_name}' not found in path tuple")
                relative_path_tuple = function_location_tuple
            else:
                index = function_location_tuple.index(self.repo_name)
                relative_path_tuple = function_location_tuple[index + 1:]
            function_location_str = os.path.join(*relative_path_tuple)
            function_code = function_code['context']
            code_content_str += f'# Path:{function_location_str}\n{function_code}\n'
        return True, code_content_str          

class Submit(Command):
    
    def __init__(self):
        super().__init__()
        
    def docstring(self) -> str:
        return (
                '============= CODE SUBMIT TOOL =============\n'
                'You also have access to a code submission tool.\n'
                'This tool executes the current code in the target file and returns the results of its execution, including any unit test outcomes and feedback related to the code’s behavior.\n'
                'You can submit code using the following command:\nSUBMIT\n'
                'SUBMIT is the word SUBMIT, and it will run the code currently present in the target file. After execution, you will receive the results of any unit tests, along with diagnostic messages or errors that occurred during runtime.\n'
            )
    
    def execute_command(self, command) -> str:
        pass

class RepoBrose(Command):
    def __init__(self, args, task_id):
        super().__init__()
        self.cmd_type = "REPO-browse"
        self.task_id = task_id
        self.work_dir = args.work_dir

    def docstring(self) -> str:
        return (
            '============= REPO BROWSE TOOL =============\n',
            'You also have access to a repository browsing tool.\n',
            'This tool allows you to browse the entire code repository associated with the current task. It helps you understand the overall structure, locate files, and explore code across different modules or components.\n',
            'You can browse the repository using the following command:\nBROWSE\n',
            'BROWSE is the word BROWSE, and it will return a list of files and directories in the repository. This will be useful for understanding how different parts of the codebase are organized and where specific functionality is implemented.\n'
        )
    def execute_command(self, command) -> str:
        repo_dir = DatasetManager.code_repo_dir(self.work_dir, self.task_id)
        files = []
        for root, dirs, filenames in os.walk(repo_dir):
            if os.path.basename(root) == "__pycache__" or '.pytest_cache' in root:
                continue
            for filename in filenames:
                files.append(os.path.relpath(os.path.join(root, filename), repo_dir))
        return True, '\n'.join(files)
        
# TODO ADD RESET and ROOLBACK COMMANDS
class ActionHandler:
    
    def __init__(self, args, task_id):
        self.cmds = {'READ': Read(args, task_id),
                    #  'EDIT': Edit(args, task_id),
                     'RETRIEVE': Retrieve(args, task_id),
                     'REPLACE': Replace(args, task_id),
                     'SUBMIT': Submit(),
                     'BROWSE': RepoBrose(args, task_id)}
    
    def command_descriptions(self):
        return '\n'.join([''.join(cmd.docstring()) for cmd in self.cmds.values()])
    
    def parse_response(self, model_resp):
        """
        Parses a command block and extracts the command type and parameters.

        Args:
            command_text (str): Raw command string (e.g., REPLACE\n<code>, EDIT N M\n<code>, etc.)

        Returns:
            dict: {
                'command_type': 'REPLACE' | 'EDIT' | 'READ' | 'RETRIEVE | 'SUBMIT',
                'parameters': dict of arguments depending on the command type
            }
        """
        lines = model_resp.strip().strip('`').splitlines()
        if not lines:
            return False, None

        # Extract command and optional arguments
        header = lines[0].strip()
        match = re.match(r'^(REPLACE|EDIT|READ|RETRIEVE|SUBMIT|BROWSE|REFLECT)(?:\s+(\d+)\s+(\d+)|\s+(.*))?$', header)
        if not match:
            return False, None

        command_type = match.group(1)
        code_lines = lines[1:]

        if command_type == "REPLACE":
            return True, {
                "command_type": "REPLACE",
                "parameters": {
                    "code": '\n'.join(code_lines).rstrip()
                }
            }
        elif command_type == "EDIT":
            start = int(match.group(2))
            end = int(match.group(3))
            return True, {
                "command_type": "EDIT",
                "parameters": {
                    "start": start,
                    "end": end,
                    "code": '\n'.join(code_lines)
                }
            }
        elif command_type == "RETRIEVE":
            query = match.group(4) if match.group(4) else '\n'.join(code_lines).strip()
            return True, {
                "command_type": "RETRIEVE",
                "parameters": {
                    "query": query
                }
            }
        elif command_type == "READ":
            file_path = match.group(4) if match.group(4) else '\n'.join(code_lines).strip()
            return True, {
                "command_type": "READ",
                "parameters": {
                    "file_path": file_path
                }
            }
        elif command_type == "SUBMIT":
            return True, {
                "command_type": "SUBMIT",
                "parameters": {}
            }   
        elif command_type == "BROWSE":
            return True, {
                "command_type": "BROWSE",
                "parameters": {
                }
            }     
    
    def extract_action(self, response: str) -> str:
        """
        Extracts the content that follows the 'action:' label in a response,
        preserving any formatting like triple backticks.

        Parameters:
            response (str): The full response string.

        Returns:
            str: The content immediately after 'action:', or an empty string if not found.
        """
        match = re.search(r'action:?\s*([\s\S]*?)(?:\n\w+:|$)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
        
    def process_command(self, model_resp):
        extract_action = self.extract_action(model_resp)
        if extract_action is None:
            return False, f'No action found in the response, plase follow the following response formate:{RESPONSE_FORMAT} !!!!!!!!!!', False
        if_success, parse_result = self.parse_response(extract_action)
        if not if_success:
            return False, 'Command syntex not correct', False
        if parse_result['command_type'] == 'SUBMIT':
            return True, None, True
        
        
        if_success, command_response = self.cmds[parse_result['command_type']].execute_command(parse_result)
        return if_success, command_response, False
    
    
    
    

    
    
    