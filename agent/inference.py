import openai
import time, tiktoken
from openai import OpenAI
import os, anthropic, json
from openai import AzureOpenAI
import logging
from openai import OpenAIError, RateLimitError, APIError, APIConnectionError, APITimeoutError
import requests
from google import genai
from google.genai import types
from together import Together

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "o3-mini": 1.10 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "o3-mini": 4.40 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])


def query_model(model_str, prompt, system_prompt, azure_api_key=None, openai_api_key=None, gemini_api_key=None,  anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    preloaded_api = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "gemini-2.0-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "gemini-1.5-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "o3-mini":
                model_str = "o3-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o3-mini-2025-01-31", messages=messages)
                answer = completion.choices[0].message.content

            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content

            try:
                if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", "o3-mini"]:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                elif model_str in ["deepseek-chat"]:
                    encoding = tiktoken.encoding_for_model("cl100k_base")
                else:
                    encoding = tiktoken.encoding_for_model(model_str)
                if model_str not in TOKENS_IN:
                    TOKENS_IN[model_str] = 0
                    TOKENS_OUT[model_str] = 0
                TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
                TOKENS_OUT[model_str] += len(encoding.encode(answer))
                if print_cost:
                    print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            except Exception as e:
                if print_cost: print(f"Cost approximation has an error? {e}")
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")

def init_feedback_result():
    return {
        "execution_result": None,
        "input_token": [],
        "output_token": [],
        'interaction_history': [],
    }

def init_code_result(taks_id, turn_number):
    return {
        "task_id": taks_id,
        "turn_number": turn_number,
        "input_token": [],
        "output_token": [],
        "prompt_cache_hit_tokens": [],
        "prompt_cache_miss_tokens": [],
        "final_code": '',
        "interaction_history": [],
        'memory': []
    }




class LLMAPIWrapper:
    
    def __init__(self, args, initial_sys_prompt=None):
        self.api_key = args.api_key
        self.api_provider = args.api_provider
        self.model = args.model
        self.deployment = args.deployment
        self.endpoint = args.endpoint
        self.api_version = args.api_version
        self.history = list()
        self.history.append({"role": "system", "content": initial_sys_prompt})
        
    def inference_agent(self, new_input, generation_result):
        if self.api_provider == 'azure-openai':
            return self.inference_agent_azureopenai(new_input, generation_result)
        raise ValueError(f"Unsupported api provider {self.api_provider}")
    
    
    def inference_agent_azureopenai(self, new_input, generation_result):
        self.history.append({"role": "user", "content": new_input})
        client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key
        )
        response = client.chat.completions.create(
            messages=self.history,
            model=self.deployment
        )
        
        response_content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": response_content})
        generation_result['input_token'].append(response.usage.prompt_tokens)
        generation_result['output_token'].append(response.usage.completion_tokens)
        generation_result['interaction_history'].append({"role": "user", "content": new_input})
        generation_result['interaction_history'].append({"role": "assistant", "content": response_content})
        return response_content
    
    def query_llm(self, sysprompt, user_prompt, query_statistics):
        if self.api_provider == 'azure-openai':
            return self.query_azure_openai(sysprompt, user_prompt, query_statistics)
        elif self.api_provider == 'openai':
            return self.query_openai(sysprompt, user_prompt, query_statistics)
        elif self.api_provider == 'deepseek':
            return self.query_deepseek(sysprompt, user_prompt, query_statistics)
        elif self.api_provider == 'google':
            return self.query_gemini(sysprompt, user_prompt, query_statistics)
        elif self.api_provider == 'together':
            return self.query_together(sysprompt, user_prompt, query_statistics)
        elif self.api_provider == 'anthropic':
            return self.query_anthropic(sysprompt, user_prompt, query_statistics)
        raise ValueError(f"Unsupported api provider {self.api_provider}")
    
    def query_anthropic(self, sysprompt, user_prompt, query_statistics, max_retries=3, delay=2.0):
        client = anthropic.Anthropic()
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 to include the initial attempt
            try:
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    system = sysprompt,
                    max_tokens=4096,
                    messages=messages,
                    temperature=0,
                    top_p=1
                )
                # If successful, update statistics and return
                query_statistics['input_token'].append(message.usage.input_tokens)
                query_statistics['output_token'].append(message.usage.output_tokens)
                query_statistics['prompt_cache_hit_tokens'].append(message.usage.cache_read_input_tokens)
                query_statistics['prompt_cache_miss_tokens'].append(None)
                query_statistics['interaction_history'].append({"role": "user", "content": user_prompt})
                query_statistics['interaction_history'].append({"role": "assistant", "content": message.content[0].text})
                
                return message.content[0].text
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"All retry attempts failed. Last error: {e}")
        
        # If we get here, all retries failed
        raise last_exception
        
    
    def query_together(self, sysprompt, user_prompt, query_statistics, max_retries=3, delay=2.0):
        client = Together()
        messages = [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": user_prompt}
        ]
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 to include the initial attempt
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    top_p=1
                )
                
                # If successful, update statistics and return
                query_statistics['input_token'].append(response.usage.prompt_tokens)
                query_statistics['output_token'].append(response.usage.completion_tokens)
                query_statistics['prompt_cache_hit_tokens'].append(response.usage.cached_tokens)
                query_statistics['prompt_cache_miss_tokens'].append(None)
                query_statistics['interaction_history'].append({"role": "user", "content": user_prompt})
                query_statistics['interaction_history'].append({"role": "assistant", "content": response.choices[0].message.content})
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"All retry attempts failed. Last error: {e}")
        
        # If we get here, all retries failed
        raise last_exception
        
    
    
    def query_gemini(self, sysprompt, user_prompt, query_statistics, max_retries=3, delay=10):
        client = genai.Client(api_key=self.api_key)
        model_thinking_budget = 1000 if self.model == 'gemini-2.5-pro' else 0
        for attempt in range(max_retries + 1):  # +1 to include the initial attempt
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=sysprompt + '\n' + user_prompt, 
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=model_thinking_budget), # Disables thinking
                        temperature=0,
                        topP=1
                    )
                )
                # If successful, update statistics and return
                query_statistics['input_token'].append(response.usage_metadata.prompt_token_count)
                query_statistics['output_token'].append(response.usage_metadata.candidates_token_count)
                query_statistics['prompt_cache_hit_tokens'].append(0 if not hasattr(response.usage_metadata, "cached_content_token_count") else response.usage_metadata.cached_content_token_count)
                query_statistics['prompt_cache_miss_tokens'].append(0)
                query_statistics['interaction_history'].append({"role": "user", "content": user_prompt})
                query_statistics['interaction_history'].append({"role": "assistant", "content": response.text})
                
                return response.text
            
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"All retry attempts failed. Last error: {e}")
        # If we get here, all retries failed
        raise last_exception
        
        
    def query_deepseek(self, sysprompt, user_prompt, query_statistics, max_retries=3, delay=2.0):
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
        # Prepare messages based on model type
        
        messages = [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": user_prompt}
        ]
        
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 to include the initial attempt
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    temperature=0,
                    top_p=1
                )
                
                # If successful, update statistics and return
                query_statistics['input_token'].append(response.usage.prompt_tokens)
                query_statistics['output_token'].append(response.usage.completion_tokens)
                query_statistics['prompt_cache_hit_tokens'].append(response.usage.prompt_cache_hit_tokens)
                query_statistics['prompt_cache_miss_tokens'].append(response.usage.prompt_cache_miss_tokens)
                query_statistics['interaction_history'].append({"role": "user", "content": user_prompt})
                query_statistics['interaction_history'].append({"role": "assistant", "content": response.choices[0].message.content})
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"All retry attempts failed. Last error: {e}")
        
        # If we get here, all retries failed
        raise last_exception
    

    def query_llm_parse(self, sysprompt, user_prompt, response_format, query_statistics):
        if self.api_provider == 'openai':
            return self.query_openai_parse(sysprompt, user_prompt, response_format, query_statistics)
        raise ValueError(f"Unsupported api provider {self.api_provider}")
        

    def query_openai_parse_no_stat(self, sysprompt, user_prompt, response_format, max_retries=3, delay=2.0):
        """
        Query OpenAI API with parse response and simple retry logic.
        
        Args:
            sysprompt: System prompt for the model
            user_prompt: User prompt for the model
            response_format: Response format for parsing
            query_statistics: Dictionary to store query statistics
            max_retries: Maximum number of retry attempts (default: 3)
            delay: Delay in seconds between retries (default: 2.0)
        
        Returns:
            Parsed response from OpenAI
            
        Raises:
            Exception: If all retry attempts fail
        """
        client = OpenAI(api_key=self.api_key)
        
        # Prepare messages based on model type
        if self.model == 'o1-mini':
            messages = [{"role": "user", "content": f'{sysprompt}\n{user_prompt}'}]
        else:
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user_prompt}
            ]
        
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 to include the initial attempt
            try:
                response = client.responses.parse(
                    model=self.model,
                    input=messages,
                    text_format=response_format,
                    # temperature=0,
                    top_p=1
                )
                
                return response.output_parsed
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"API parse call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"All retry attempts failed. Last error: {e}")
        
        # If we get here, all retries failed
        raise last_exception
    

    def query_openai_parse(self, sysprompt, user_prompt, response_format, query_statistics, max_retries=3, delay=2.0):
        """
        Query OpenAI API with parse response and simple retry logic.
        
        Args:
            sysprompt: System prompt for the model
            user_prompt: User prompt for the model
            response_format: Response format for parsing
            query_statistics: Dictionary to store query statistics
            max_retries: Maximum number of retry attempts (default: 3)
            delay: Delay in seconds between retries (default: 2.0)
        
        Returns:
            Parsed response from OpenAI
            
        Raises:
            Exception: If all retry attempts fail
        """
        client = OpenAI(api_key=self.api_key)
        
        # Prepare messages based on model type
        if self.model == 'o1-mini':
            messages = [{"role": "user", "content": f'{sysprompt}\n{user_prompt}'}]
        else:
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user_prompt}
            ]
        
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 to include the initial attempt
            try:
                response = client.responses.parse(
                    model=self.model,
                    input=messages,
                    text_format=response_format,
                    # temperature=0,
                    top_p=1
                )
                
                # If successful, update statistics and return
                query_statistics['input_token'].append(response.usage.input_tokens)
                query_statistics['output_token'].append(response.usage.output_tokens)
                query_statistics['interaction_history'].append({"role": "user", "content": user_prompt})
                
                return response.output_parsed
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"API parse call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"All retry attempts failed. Last error: {e}")
        
        # If we get here, all retries failed
        raise last_exception
    
    def query_openai(self, sysprompt, user_prompt, query_statistics, max_retries=3, delay=2.0):
        """
        Query OpenAI API with simple retry logic.
        
        Args:
            sysprompt: System prompt for the model
            user_prompt: User prompt for the model
            query_statistics: Dictionary to store query statistics
            max_retries: Maximum number of retry attempts (default: 3)
            delay: Delay in seconds between retries (default: 2.0)
        
        Returns:
            str: The response content from OpenAI
            
        Raises:
            Exception: If all retry attempts fail
        """
        client = OpenAI(api_key=self.api_key)
        
        # Prepare messages based on model type
        if self.model == 'o1-mini':
            messages = [{"role": "user", "content": f'{sysprompt}\n{user_prompt}'}]
        else:
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user_prompt}
            ]
        
        last_exception = None
        
        for attempt in range(max_retries + 1):  # +1 to include the initial attempt
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # temperature=0,
                    top_p=1
                )
                
                # If successful, update statistics and return
                query_statistics['input_token'].append(response.usage.prompt_tokens)
                query_statistics['output_token'].append(response.usage.completion_tokens)
                query_statistics['prompt_cache_hit_tokens'].append(response.usage.prompt_tokens_details.cached_tokens)
                query_statistics['prompt_cache_miss_tokens'].append(None)
                query_statistics['interaction_history'].append({"role": "user", "content": user_prompt})
                query_statistics['interaction_history'].append({"role": "assistant", "content": response.choices[0].message.content})
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"All retry attempts failed. Last error: {e}")
        
        # If we get here, all retries failed
        raise last_exception
        
    def query_azure_openai(self, sysprompt, user_prompt, query_statistics):
        client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )
        if self.model == 'o1-mini':
            messages = [{"role": "user", "content": f'{sysprompt}\n{user_prompt}'}]
        else:
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user_prompt}
            ]
        response = client.chat.completions.create(
            model=self.deployment,  # Use the deployment name, not the base model name
            messages=messages,
            temperature=0,
            top_p=1
        )
        
        query_statistics['input_token'].append(response.usage.prompt_tokens)
        query_statistics['output_token'].append(response.usage.completion_tokens)
        query_statistics['interaction_history'].append({"role": "user", "content": user_prompt})
        query_statistics['interaction_history'].append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
    
    def reset_history(self, initial_sys_prompt):
        self.history = []
        self.history.append({"role": "system", "content": initial_sys_prompt})
        