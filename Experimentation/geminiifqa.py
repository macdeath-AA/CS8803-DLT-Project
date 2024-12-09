import google.generativeai as genai
import pandas as pd

genai.configure(api_key="yourkey")
model = genai.GenerativeModel("gemini-1.5-flash")

filename = 'ifQA_processed.csv'

df = pd.read_csv(filename)
context = df['context']
questions = df['question']
prompts = df['context'] + df['question']

responses = []
for prompt in prompts:
    mod_prompt = f"{prompt} Keep your answer short."
    response = model.generate_content(mod_prompt, request_options={'timeout':300})
    responses.append(response.text)

df['response'] = responses

output_file = 'geminianswers.csv'
df.to_csv(output_file, index=False)