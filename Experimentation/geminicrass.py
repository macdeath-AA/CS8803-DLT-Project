import google.generativeai as genai
import pandas as pd
import time
from google.api_core import exceptions 
import os

genai.configure(api_key="yourkey")
model = genai.GenerativeModel("gemini-1.5-flash")

filename = 'ifQA_processed.csv'

df = pd.read_csv(filename)
context = df['context']
questions = df['question']
answers = df['answers']
prompts = df['context'] + df['question']

output_filename = 'geminianswers_final.csv'

if not os.path.exists(output_filename):
    # If it doesn't exist, create a new DataFrame with appropriate columns and save it
    df_output = pd.DataFrame(columns=['context', 'question', 'answers', 'response'])
    df_output.to_csv(output_filename, index=False)

responses = []

for index, prompt in enumerate(prompts):
    mod_prompt = f"{prompt} Keep your answer short."
    try:
        response = model.generate_content(mod_prompt, request_options={'timeout':300})
        # responses.append(response.text)
        # print(response.text)
        response_text = response.text 

    except exceptions.ResourceExhausted as e:
        print("Resource Exhausted. Waiting...")
        time.sleep(60)
        response = model.generate_content(mod_prompt, request_options={'timeout':300})
        # responses.append(response.text)
        # print(response.text)
        response_text = response.text
    
    except (UnicodeEncodeError, UnicodeDecodeError, UnicodeError) as ue:
        print(f"Unicode Error: {ue}")
        # responses.append("Unicode Error")
        response_text = "Unicode Error"
    
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
        # responses.append("Unexpected error")
        response_text = "Unexpected error"
    
    row_data = {
        'context': df['context'][index],
        'question': df['question'][index],
        'answers' : df['answers'][index],
        'response': response_text
    }

    row_df = pd.DataFrame([row_data])
    row_df.to_csv(output_filename, mode='a', header=False, index=False)

    time.sleep(10)

print('hope its over')