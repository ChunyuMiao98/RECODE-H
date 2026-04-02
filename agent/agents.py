from utils import *
from tools import *
from inference import *
from constants import *
import random, string
import re
import json
import subprocess
from action import ActionHandler
from pydantic import BaseModel

# Clean runner that avoids `conda activate` + `cp/rm` mutation of target repos.
from clean_test_runner import run_annotation_tests

class Arg:
    pass



class FeedbackInstance(BaseModel):
    interface: str
    category: str
    description: str
    analysis: str
    actionable_feedback: str
    direct_code_feedback: str
    
# class FeedbackInstanceNew(BaseModel):
#     interface: str
#     error_category: str
#     l1_summary: str
#     incorrect_code_snippet: str
#     l2_justification: str
#     l3_implementation_guidance: str
#     l4_explicit_correction: str
    
class Feedbacks(BaseModel):
    feedbacks: list[FeedbackInstance]
    
class Difference(BaseModel):
    generated_function_name: str
    canonical_function_name: str
    generated_implementation_description: str
    canonical_implementation_description: str
    difference_description: str

class Differences(BaseModel):
    feedbacks: list[Difference]



def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found

def format_feedback(feedbacks, guidance_level, guidance_type):
    level_feedback = ''
    if guidance_type == 'fixed':
        for i in range(len(feedbacks)):
            # if guidance_level > 0:
            #     level_feedback += f'Feedback {i}:\nInterface:{feedbacks[i]["interface"]}\n'
            #     level_feedback = level_feedback + f'Summary: {feedbacks[i]["l1_summary"]}\n'
            #     level_feedback = level_feedback + f'Incorrect code snippet: {feedbacks[i]["incorrect_code_snippet"]}\n'
            # if guidance_level > 1:
            #     # level_feedback = level_feedback + f'Incorrect Code Snippet: {feedbacks[i]["incorrect_code_snippet"]}\n'
            #     level_feedback = level_feedback + f'Analysis: {feedbacks[i]["l2_justification"]}\n'
            # if guidance_level > 2:
            #     level_feedback = level_feedback + f'Implementation Guidance: {feedbacks[i]["l3_implementation_guidance"]}\n'
            # if guidance_level > 3:
            #     level_feedback = level_feedback + f'Explicit correction: {feedbacks[i]["l4_explicit_correction"]}\n'
            if guidance_level > 0:
                level_feedback += f'Feedback {i}:\nInterface:{feedbacks[i]["interface"]}\n'
                level_feedback = level_feedback + f'Description: {feedbacks[i]["description"]}\n'
            if guidance_level > 1:
                # level_feedback = level_feedback + f'Incorrect Code Snippet: {feedbacks[i]["incorrect_code_snippet"]}\n'
                level_feedback = level_feedback + f'Analysis: {feedbacks[i]["analysis"]}\n'
            if guidance_level > 2:
                level_feedback = level_feedback + f'Actionable Feedback: {feedbacks[i]["actionable_feedback"]}\n'
            if guidance_level > 3:
                level_feedback = level_feedback + f'Direct code feedback: {feedbacks[i]["direct_code_feedback"]}\n'
            if guidance_level not in [0, 1, 2, 3, 4]:
                raise ValueError(f"Unknown guidance level: {guidance_level}. Supported levels are 0, 1, 2, 3, and 4.")
    else:
        raise ValueError("Other type of guidance is not implemented")
    if level_feedback == '':
        level_feedback = 'No feedback'
    return level_feedback
        
# TODO: Replay
class CodeAgent:
    def __init__(self, args):
        self.max_step = args.max_steps
        self.work_dir = args.work_dir
        self.args = args
        
        self.memory_threshold = getattr(args, 'memory_threshold', 10) # Trigger summarization when history > 8 messages
        # <<< CHANGE: Define how many recent messages to preserve verbatim >>>
        self.memory_to_keep = getattr(args, 'messages_to_keep', 5) # Keep the last 4 messages (2 turns)
        
    def final_revise(self):
        pass
    
    def set_history(self, history):
        self.history = history
             

    def update_task_id(self, task_id):
        """
        Update the task ID for the code agent.
        
        Parameters:
            task_id (str): The new task ID to set.
        """
        self.task_id = task_id
        self.action_handler = ActionHandler(self.args, task_id)
        self.llm_api_wrapper = LLMAPIWrapper(self.args, self.build_system_prompt())
        
        # <<< CHANGE: Initialize an explicit history log >>>
        # This list will store dictionaries, e.g., {'role': 'user', 'content': '...'}
        self.history = []
        
    def _summarize_history(self, memory_keep_num):
        """
        Summarizes the OLD part of the conversation history, avoiding redundancy
        with the recent turns that are kept verbatim.
        """
        
        if not self.history:
            return
        if memory_keep_num == 0:
            self.history = []
            return 
        # print("--- Managing Memory: Summarizing History ---")
        if memory_keep_num is not None:
            history_to_summarize = self.history[:-memory_keep_num]
            recent_history = self.history[-memory_keep_num:]
        # <<< CHANGE: Split history into what to summarize and what to keep >>>
        else:
            history_to_summarize = self.history[:-self.memory_to_keep]
            recent_history = self.history[-self.memory_to_keep:]

        # If there's nothing old enough to summarize, do nothing.
        if not history_to_summarize:
            return
        summary_stat = init_code_result(self.task_id, 0)
        conversation_to_summarize_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_to_summarize])
        summary = self.llm_api_wrapper.query_llm("You are a helpful summarization assistant.", SUMMARIZATION_PROMP.format(conversation_to_summarize_str), summary_stat)
        
        # <<< CHANGE: Construct the new history without redundancy >>>
        # The new history is the summary of the old part + the recent part verbatim.
        self.history = [
            {'role': 'system', 'content': f"Summary of earlier work:\n{summary}"}
        ] + recent_history
        return summary_stat
        # print("--- Memory Summarization Complete ---")

    # We also need to update _manage_memory to use the new threshold variable correctly
    def _manage_memory(self, memory_keep_num):
        """Checks if the history is too long and triggers summarization."""
        if len(self.history) > self.memory_threshold:
            return self._summarize_history(None)
        elif memory_keep_num is not None:
            return self._summarize_history(memory_keep_num)
        
    def generate_llm(self, feedback):
        turn_result = init_feedback_result()
        turn_result['task_id'] = self.task_id
        sys_prompt = self.build_llm_system_prompt()
        user_prompt = self.build_llm_user_prompt(feedback)
        model_resp = self.llm_api_wrapper.query_llm(sys_prompt, user_prompt, turn_result)
        code = self.extract_code_from_response(model_resp)
        if code:
            # Save the code to the file
            DatasetManager.append_new_code_content(self.work_dir, self.task_id, code)
            turn_result['final_code'] = code
        
        return turn_result
        # code = self.clean_response(model_resp)
    
    def extract_code_from_response(self, response: str) -> str:
        """
        Extracts code blocks from a raw LLM response.

        Parameters:
            response (str): The raw text response from the LLM.

        Returns:
            str: The extracted code block(s), concatenated as a single string.
        """
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        return "\n\n".join(code_blocks).strip()
    
    
    def generate_agent_history(self, feedback=None, result=None, turn_num=0):
        """
        Generates code through a multi-turn conversation with explicit, self-managed memory.
        """
        generation_result = init_code_result(self.task_id, turn_num)
        
        if feedback is None:
            feedback = 'No feedback yet'
        else:
            feedback = self.process_feedback(feedback)
        if result is None:
            result = 'No Action execution result'
        else:
            test_info = build_test_info(result)
            result = self.build_next_response(test_info)
            # if result['errors'] == 0:

        for i in range(self.args.max_steps):
            print(i)
            summary_stat = self._manage_memory(None)
            
            # 2. Call a stateless LLM endpoint (query_llm)
            for retry_time in range(3):
                try:
                    full_user_prompt = self._build_full_user_prompt(result, feedback, self.args.max_steps-i)
                    model_resp = self.llm_api_wrapper.query_llm(self.build_system_prompt(), full_user_prompt, generation_result)
                    break
                except Exception as e:
                    if retry_time == 2:
                        raise e
                    else:
                        self._manage_memory(0)
            model_resp = self.clean_text(model_resp)
            # 3. Update our explicit history log
            self.history.append({'role': 'user', 'content': result})
            self.history.append({'role': 'assistant', 'content': model_resp})

            # 4. Process the response
            if_success, result, if_submit = self.action_handler.process_command(model_resp)

            if if_submit:
                generation_result['final_code'] = DatasetManager.load_code_raw_content(self.work_dir, self.task_id)
                generation_result['memory'] = self.history
                return generation_result
            
            # 5. Prepare the prompt for the next iteration
            result = self.build_invalide_command_response() if not if_success else self.build_next_response(result)

        generation_result['final_code'] = DatasetManager.load_code_raw_content(self.work_dir, self.task_id)
        generation_result['memory'] = self.history
        return generation_result


    def generate_conversition_history(self, feedback=None, turn_number=0):
        generation_result = init_code_result(self.task_id, turn_number)
        cur_response = self.build_conversation_init_response(feedback)
        for i in range(self.max_step):
            # remaining_step = self.max_step - i
            model_resp = self.llm_api_wrapper.inference_agent(cur_response, generation_result)
            model_resp = self.clean_text(model_resp)
            if_success, result, if_submit = self.action_handler.process_command(model_resp)
            if if_submit:
                generation_result['final_code'] = DatasetManager.load_code_raw_content(self.work_dir, self.task_id)
                return generation_result
            elif not if_success:
                cur_response = self.build_invalide_command_response()
            else:
                cur_response = self.build_next_response(result)
        generation_result['final_code'] = DatasetManager.load_code_raw_content(self.work_dir, self.task_id)
        return generation_result
        # For now No final revise is applied
        # return self.final_revise()

    def build_invalide_command_response(self):
        '''
        Build the response when the command is invalid.
        '''
        return 'Invalid command, Please follow the correct syntex to use the tool\nImportant:\nYou should follow this format:\n```\nreflect: [Your reflection on the context and the action you are going to take]\naction: [The action you are going to take]\n```'  

    def build_next_response(self, command_result):
        return 'Command result:\n' + f'{command_result}\n' + GENERATION_GUIDANCE_C
        
    def build_conversation_init_response(self, feedback):
        if feedback == None:
            return self.build_initial_task_prompt()
        else:
            return  self.code_content_prompt() + self.process_feedback(feedback)
    
    def code_content_prompt(self):
        return '# Current code content: \n'+  DatasetManager.load_code_content(self.work_dir, self.task_id)
    
    def process_feedback(self, feedback):
        return '\n# Feedback: \n' + feedback
        
    def build_latex_content(self):
        '''read the content of latex file and return'''
        task_latex_content = DatasetManager.load_task_latex(self.work_dir, self.task_id)
        return task_latex_content
    
    def build_code_generation_instruction(self):
        '''read the code generation instruction file and return'''
        task_instructin = DatasetManager.load_task_instruction(self.work_dir, self.task_id)
        return task_instructin
    
    def build_history(self):
        """Builds a concise history string from the managed history list for the new method."""
        if not self.history:
            return "No history yet."
        return "\n\n".join([f"### {msg['role'].capitalize()}\n{msg['content']}" for msg in self.history])
    
    def build_feedback(self, feedback):
        ''' Buld concise feedback string based on the input feedback'''
        return feedback
    
    def target_file_path(self):
        '''
        Return the target file path of the code generation task.
        The target file is the file that the code agent will generate code into.
        '''
        return DatasetManager.target_file_path(self.work_dir, self.task_id)
    
    def build_current_code(self):
        '''
        Read the code file and return the contents?
        Optinal: Read the code file and extract the target function part and retrun.
        '''
        current_code = DatasetManager.load_code_content(self.work_dir, self.task_id)
        if current_code.strip() == '':
            current_code = '# Currently the code file is empty'
        return current_code
    
    def _build_full_user_prompt(self, current_response: str, feedback: str, i):
        """Constructs the complete user prompt for the LLM for the agent-history method."""
        return CODE_AGENT_USER_PROMPT.format(
            self.build_latex_content(),
            self.build_code_generation_instruction(),
            self.build_history(),
            self.build_current_code(),
            feedback,
            current_response,
            GENERATION_GUIDANCE_C + f'\nYou have {i} step before autometic submit. You need generate all required interfaces when submit.'
        )
        
    def build_initial_task_prompt(self):
        '''
        Task prompt of initial step include:
        1. Latex code of relevent content
        2. Code generation instruction
        3. Current Code Implementation
        4. Generation guidance
        '''
        return CODE_AGENT_INITIAL_PROMPT.format(self.build_latex_content(), 
                                                self.build_code_generation_instruction(),
                                                self.target_file_path(),
                                                self.build_current_code(),
                                                GENERATION_GUIDANCE_C
                                                )

    def build_llm_user_prompt(self, feedback):
        '''
        Task prompt of initial step include:
        1. Latex code of relevent content
        2. Code generation instruction
        3. Current Code Implementation
        4. Generation guidance
        '''
        return LLM_BASELINE_USER_PROMPT.format(self.build_latex_content(), 
                                                self.build_code_generation_instruction(),
                                                self.target_file_path(),
                                                self.build_current_code(),
                                                feedback if feedback else 'No feedback provided',
                                                BASELINE_GENERATION_GUIDANCE
                                                )
        

    def build_llm_system_prompt(self):
        return LLM_BASELINE_SYS_PROMPT
    
    def build_system_prompt(self):
        '''
        Retrun the system prompt to the code agent, including the following information:
        1. Task description
        2. Input description
        3. Output description
        4. Action description
        '''
        return CODE_AGENT_SYS_PROMPT.format(self.action_handler.command_descriptions())
    
    @staticmethod
    def clean_text(text):
        """
        Fix minor corrections
        :return: (str) corrected text
        """
        text = text.replace("```\n", "```")
        return text
        
        
class HumanAgent:
    def __init__(self, args):
        self.work_dir = args.work_dir
        self.guidance_type = args.guidance_type
        self.guidance_level = args.guidance_level
        self.api_wrapper = self.build_api_wrapper(args)
        self.history = list()
    
    def update_task_id(self, task_id):
        self.task_id = task_id


    def build_api_wrapper(self, args):
        new_args = Arg
        new_args.api_key = args.human_api_key
        new_args.api_provider = args.human_api_provider
        new_args.model = args.human_agent_model
        new_args.deployment = args.human_deployment
        new_args.endpoint = args.human_endpoint
        new_args.api_version = args.human_api_version
        return LLMAPIWrapper(new_args)
        
    def execute_feedback(self, if_feedback):
        print('if_feedback:', if_feedback)
        human_agent_result = init_feedback_result()
        # Run unit tests for the current task (writes junitxml + runner logs under annotation dir).
        run_annotation_tests(self.work_dir, self.task_id, timeout_sec=600)
        parsed_result = analyze_pytest_xml(self.work_dir, self.task_id)
        human_agent_result['execution_result'] = parsed_result
        if parsed_result['errors'] == 0 and parsed_result['failures'] == 0 and parsed_result['skipped'] == 0:
            return human_agent_result, True, None
        elif if_feedback and (self.guidance_type == 'fixed' and self.guidance_level > 0):
            print(f"To generate feedback")
            feedback = self.generate_feedback(parsed_result, human_agent_result)
            
            return human_agent_result, False, feedback
        else:
            return human_agent_result, False, None
    
    def execute_feedback_raw(self, if_feedback: bool):
        """
        Same contract as execute_feedback, but when feedback is generated,
        returns the *unformatted* raw feedback list (dicts), suitable to save as a seed.
        """
        print('if_feedback:', if_feedback)
        human_agent_result = init_feedback_result()
        # Run unit tests for the current task (writes junitxml + runner logs under annotation dir).
        run_annotation_tests(self.work_dir, self.task_id, timeout_sec=600)
        parsed_result = analyze_pytest_xml(self.work_dir, self.task_id)
        human_agent_result['execution_result'] = parsed_result
        if parsed_result['errors'] == 0 and parsed_result['failures'] == 0 and parsed_result['skipped'] == 0:
            return human_agent_result, True, None
        elif if_feedback and (self.guidance_type == 'fixed' and self.guidance_level > 0):
            print(f"To generate feedback")
            feedback = self.generate_feedback(parsed_result, human_agent_result)
            
            return human_agent_result, False, feedback
        else:
            return human_agent_result, False, None
    
    def generate_feedback_pipline(self, parsed_result, human_agent_result):
        '''
        Generate the feedback with a pipeline:
        1. Difference Identification
        2. Feedback generate and format
        '''
        
        # Step 1: Difference Identification
        error_info = build_test_info(parsed_result)
        
        # Build prompts for difference identification
        diff_sys_prompt = self.build_difference_identification_sys_prompt()
        diff_user_prompt = self.build_difference_identification_user_prompt(error_info)
        
        # Query LLM for difference identification
        differences = self.api_wrapper.query_llm_parse(diff_sys_prompt, diff_user_prompt, Differences, human_agent_result)
        
        # Log the difference identification step
        human_agent_result['interaction_history'].append({
            "role": "system", 
            "content": f"Difference Identification: {differences}"
        })
        
        # Step 2: Feedback Generation and Formatting
        if self.guidance_type == 'fixed':
            feedback_sys_prompt = self.build_feedback_sys_prompt_constant_pipline()
            feedback_user_prompt = self.build_feedback_user_prompt_with_differences(error_info, differences)
            
        elif self.guidance_type == 'dynamic':
            feedback_sys_prompt = self.build_feedback_sys_prompt_dynamic_pipline()
            feedback_user_prompt = self.build_feedback_user_prompt_dynamic_with_differences(error_info, differences)
        
        # Generate structured feedback
        feedbacks = self.api_wrapper.query_llm_parse(feedback_sys_prompt, feedback_user_prompt, Feedbacks, human_agent_result)
        
        # Parse and format the feedback
        feedbacks = self.parse_feedback_dic(feedbacks)
        human_agent_result['interaction_history'].append({"role": "assistant", "content": feedbacks})
        
        # Format according to guidance level
        formatted_feedback = self.parse_feedback(feedbacks)
        
        return formatted_feedback
    
    def build_feedback_sys_prompt_constant_pipline(self):
        return HUMAN_AGENT_SYS_PROMPT_CONSTANT_PIPLINE
        
    def build_feedback_sys_prompt_dynamic_pipline(self):
        pass
    
    def build_difference_identification_sys_prompt(self):
        '''
        System prompt for the difference identification step
        '''
        return HUMAN_AGENT_SYS_IDENTIFY_DIFFERENCE

    def build_difference_identification_user_prompt(self, error_info):
        '''
        User prompt for difference identification including all necessary context
        '''
        return f"""# Task Context
    ## LaTeX Description:
    {DatasetManager.load_task_latex(self.work_dir, self.task_id)}

    ## Implementation Instructions:
    {DatasetManager.load_task_instruction(self.work_dir, self.task_id)}

    ## Expected (Canonical) Code:
    ```python
    {DatasetManager.load_task_canonical(self.work_dir, self.task_id)}
    ```

    ## Generated Code:
    ```python
    {DatasetManager.load_code_raw_content(self.work_dir, self.task_id)}
    ```

    Please analyze the differences between the expected and generated code as required in the system prompt."""

    def build_feedback_user_prompt_with_differences(self, error_info, differences):
        '''
        Enhanced user prompt that includes identified differences for more targeted feedback
        '''
        return HUMAN_AGENT_USER_PROMPT_PIPLINE.format(
            DatasetManager.load_task_latex(self.work_dir, self.task_id),
            DatasetManager.load_task_instruction(self.work_dir, self.task_id),
            'python',
            DatasetManager.load_task_canonical(self.work_dir, self.task_id),
            'python',
            DatasetManager.load_code_raw_content(self.work_dir, self.task_id),
            error_info,
            self.formate_difference_str(differences),
            FEEDBACK_CATEGORY_INTRODUCTION_PROMPT
        )
        
    def formate_difference_str(self, differences):
        '''
        Format a difference string using the differences obj
        
        Args:
            differences (Differences): A Differences object containing a list of Difference objects
            
        Returns:
            str: A formatted string representation of the differences
        '''
        if not differences or not differences.feedbacks:
            return "No differences identified."
        
        formatted_str = "## Identified Differences:\n\n"
        
        for i, diff in enumerate(differences.feedbacks, 1):
            formatted_str += f"### Difference {i}:\n"
            formatted_str += f"**Generated Function:** `{diff.generated_function_name}`\n"
            formatted_str += f"**Canonical Function:** `{diff.canonical_function_name}`\n"
            formatted_str += f"**Generated Implementation:** {diff.generated_implementation_description}\n"
            formatted_str += f"**Canonical Implementation:** {diff.canonical_implementation_description}\n"
            formatted_str += f"**Key Difference:** {diff.difference_description}\n"
            formatted_str += "\n"
        
        return formatted_str

    def build_feedback_user_prompt_dynamic_with_differences(self, error_info, differences):
        '''
        Dynamic feedback prompt that incorporates identified differences
        '''
        # This would be implemented when dynamic guidance is supported
        # For now, fall back to the fixed approach
        return self.build_feedback_user_prompt_with_differences(error_info, differences)
    
    
    def generate_feedback(self, parsed_result, human_agent_result):
        if self.guidance_type == 'fixed':
            error_info = build_test_info(parsed_result)
            sys_prompt = self.build_feedback_sys_prompt_constant()
            user_prompt = self.build_feedback_user_prompt_constant(error_info)
            
        elif self.guidance_type == 'dynamic':
            sys_prompt = self.build_feedback_sys_prompt_dynamic()
            user_prompt = self.build_feedback_user_prompt_dynamic(error_info)

        feedbacks = self.api_wrapper.query_llm_parse(sys_prompt, user_prompt, Feedbacks, human_agent_result)
        
        feedbacks = self.parse_feedback_dic(feedbacks)
        human_agent_result['interaction_history'].append({"role": "assistant", "content": feedbacks})
        feedbacks = self.parse_feedback(feedbacks)
        return feedbacks
    
    def parse_feedback_dic(self, feedbacks):
        # return [{
        #     'interface': feedback.interface,
        #     "error_category": feedback.error_category,
        #     "l1_summary": feedback.l1_summary,
        #     "incorrect_code_snippet": feedback.incorrect_code_snippet,
        #     "l2_justification": feedback.l2_justification,
        #     "l3_implementation_guidance": feedback.l3_implementation_guidance,
        #     "l4_explicit_correction": feedback.l4_explicit_correction
        # } for feedback in feedbacks.feedbacks]
        return [{
            'interface': feedback.interface,
            "category": feedback.category,
            "description": feedback.description,
            "analysis": feedback.analysis,
            "actionable_feedback": feedback.actionable_feedback,
            "direct_code_feedback": feedback.direct_code_feedback
        } for feedback in feedbacks.feedbacks]
    
    def parse_feedback(self, feedbacks):
        """
        Parse a JSON string containing feedback data and return a list of dictionaries.
        
        Args:
            json_string (str): JSON string containing feedback data with a "differences" array
            
        Returns:
            list: List of dictionaries containing the feedback data
            
        Raises:
            json.JSONDecodeError: If the JSON string is invalid
            KeyError: If the expected "differences" key is not found
            TypeError: If the input is not a string
        """
        # Clean up the data by removing the "<thinking>" keys since they're not needed
        cleaned_differences = []
        for item in feedbacks:
            if isinstance(item, dict):
                # Create a copy without the "<thinking>" key
                cleaned_item = {k: v for k, v in item.items() if k != "<thinking>"}
                cleaned_differences.append(cleaned_item)
            else:
                cleaned_differences.append(item)
        level_feedback = format_feedback(cleaned_differences, self.guidance_level, self.guidance_type)
        return level_feedback
    
            
    def build_feedback_sys_prompt_dynamic(self):
        pass
    
    def build_feedback_user_prompt_dynamic(self, error_info):
        '''
        User Prompt contains the following information:
        1. Latex code
        2. Instruction
        3. Canonical code
        4. Generated code
        5. Execution Error information
        6. Generation guidance + Guidance level description
        '''
        pass
    
    def build_feedback_sys_prompt_constant(self):
        return HUMAN_AGENT_SYS_PROMPT_CONSTANT
    
    def build_feedback_user_prompt_constant(self, error_info):
        '''
        User Prompt contains the following information:
        1. Latex code
        2. Instruction
        3. Canonical code
        4. Generated code
        5. Execution Error information
        6. Generation guidance + Guidance level description
        '''
        # if self.guidance_type == 'fixed':
        #     if self.guidance_level == 0:
        #         guidance_feedback = L0_FEEDBACK_SPECIFICATION
        #     elif self.guidance_level == 1:
        #         guidance_feedback = L1_FEEDBACK_SPECIFICATION
        #     elif self.guidance_level == 2:
        #         guidance_feedback = L2_FEEDBACK_SPECIFICATION
        #     elif self.guidance_level == 3:
        #         guidance_feedback = L3_FEEDBACK_SPECIFICATION
        #     elif self.guidance_level == 4:
        #         guidance_feedback = L4_FEEDBACK_SPECIFICATION
        #     else:
        #         raise ValueError(f"Unknown guidance level: {self.guidance_level}. Supported levels are 0, 1, 2, 3, and 4.")
        # else:
        #     raise ValueError("Other type of guidance is not implemented")

        # return HUMAN_AGENT_USER_PROMPT.format(
        #     DatasetManager.load_task_latex(self.work_dir, self.task_id),
        #     DatasetManager.load_task_instruction(self.work_dir, self.task_id),
        #     'python',
        #     DatasetManager.load_task_canonical(self.work_dir, self.task_id),
        #     'python',
        #     DatasetManager.load_code_raw_content(self.work_dir, self.task_id),
        #     error_info,
        #     FEEDBACK_CATEGORIES,
        #     guidance_feedback
        # )
        return HUMAN_AGENT_USER_PROMPT.format(
            DatasetManager.load_task_latex(self.work_dir, self.task_id),
            DatasetManager.load_task_instruction(self.work_dir, self.task_id),
            'python',
            DatasetManager.load_task_canonical(self.work_dir, self.task_id),
            'python',
            DatasetManager.load_code_raw_content(self.work_dir, self.task_id),
            error_info,
            FEEDBACK_CATEGORY_INTRODUCTION_PROMPT
        )
        
        
        
