
import os
import traceback
os.environ["HF_HOME"] = "../hf_cache"
os.environ["HF_TOKEN"] = "your-token"

# MD5 Checksums are at: /home/hice1/rsrivastava76/.llama/checkpoints/Llama3.2-1B/checklist.chk

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from utils.util import extract_first_sentences
from utils.loggers import loggers
from tenacity import wait_random_exponential, stop_after_attempt, retry, RetryError
import torch


def log_attempt_number(retry_state):
    """return the result of the last call attempt"""
    loggers["error"].info(f"Retrying: attempt #{retry_state.attempt_number}, wait: {retry_state.outcome_timestamp} for {retry_state.outcome.exception()}")

class LLM_API():
    def __init__(self, model="meta-llama/Llama-3.2-1B-Instruct", query_params=None):
        self.model = "meta-llama/Llama-3.2-1B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            load_in_8bit=True,
            torch_dtype=torch.float,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            padding_side="left",
            use_fast=True,
        )

        self.client = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        self.query_params = query_params
        self.token_usage = {"input": 0, "output": 0}


    @retry(wait=wait_random_exponential(min=2, max=8), stop=stop_after_attempt(30), after=log_attempt_number)
    def query(self,
            prompt,
            temp=None,
            n=None,
            stop=None,
            max_tokens=None,
        ):
        prompt_chat = [
                {"role": "user", "content": prompt}
            ]
        outputs = self.client(
            prompt_chat,
            do_sample=True,
            add_special_tokens=True,
            num_return_sequences=1 if n is None else n,
            max_new_tokens=self.query_params['max_tokens'] if max_tokens is None else max_tokens,
            temperature=self.query_params['temperature'] if temp is None else temp,
            stop_strings=self.query_params['stop'] if stop is None else stop,
            tokenizer = self.tokenizer
            # frequency_penalty=self.query_params['frequency_penalty'],
            # presence_penalty=self.query_params['presence_penalty'],
        )
        contents = []
#             print("Outputs")
#             print(outputs)
        for output in outputs:
            contents.append(output["generated_text"][1]["content"].strip())
#             self.token_usage['input'] += output.usage.prompt_tokens
#             self.token_usage['output'] += output.usage.completion_tokens

        if len(contents) == 0:
            raise RuntimeError(f"no response from model {self.model}")
        return contents


    def get_response(
                self,
                prompt,
                temp=None,
                n=None,
                stop=None,
                max_tokens=None,
                extract_first_sentence=True,
            ):
        try:
            if extract_first_sentence:      
                return extract_first_sentences(self.query(prompt, temp=temp, n=n, stop=stop, max_tokens=max_tokens))
            else:
                return self.query(prompt, temp=temp, n=n, stop=stop, max_tokens=max_tokens)
        except RetryError as e:
            print(e)
            traceback.print_exc()
            return '<SKIP>'


    def reset_token_usage(self):
        self.token_usage = {"input": 0, "output": 0}


    def get_token_usage(self):
        return self.token_usage