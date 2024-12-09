from algo.reasoning_adapter import Reasoning_Adapter
from utils.strategyqa_metric import get_accuracy, stop_criterion, is_correct
from ast import literal_eval
PROMPT = '''
Use the step-by-step method as shown in the examples to answer the question. Break down the problem into smaller parts and then provide the final answer after '####'. Do not use the '####' sequence if you are not giving the answer in that very sentence. '####' is an escape sequence take it seriously

Example 1:
Context: the Central Michigan Chippewas were named as the Sun Bowl replacement team. The Chippewas had originally been scheduled to face the Boise State Broncos in the Arizona Bowl, until Boise State withdrew from that bowl due to COVID-19 issues. History. The first Sun Bowl was the 1935 edition, played on New Year's Day between Texas high school teams; the 1936 edition, played one year later, was the first Sun Bowl contested between college teams. In most of its early history, the game pitted the champion of the Border Conference against an at-large opponent. The first three editions were played at

Question: If the Boise State Broncos hadn't withdrawn from the Arizona Bowl due to COVID-19 issues, which team would they have competed against?
#### ['The Chippewas', 'The Central Michigan Chippewas']

Example 2:
Context: Gatorade is an American brand of sports-themed beverage and food products, built around its signature line of sports drinks. Gatorade is currently manufactured by PepsiCo and is distributed in over 80 countries. The beverage was first developed in 1965 by a team of researchers led by Dr. Robert Cade. It was originally made for the Gators at the University of Florida to replenish the carbohydrates that the school's student-athletes burned and the combination of water and electrolytes that they lost in sweat during rigorous sports activities. Originally produced and marketed by Stokely-Van Camp, the Gatorade brand was purchased by the

Question: If Gatorade's distribution was halved, in how many countries would it be available?
#### ['40']

Your Question:
'''.strip()

PROMPT_NO_INST = '''
Only Output a Single sentence starting with '####' and the answer in one sentence. Do not output anything else
'''.strip() + "\n"
# IFQA Prompt Example:
# PROMPT = '''Context: Samuel Langhorne Clemens (November 30, 1835 â€“ April 21, 1910), known by his pen name Mark Twain, was an American writer, humorist, entrepreneur, publisher, and lecturer. He was lauded as the "greatest humorist the United States has produced," and William Faulkner called him "the father of American literature". His novels include "The Adventures of Tom Sawyer" (1876) and its sequel, "Adventures of Huckleberry Finn" (1884), the latter of which has often been called the "Great American Novel". Twain was raised in Hannibal, Missouri, which later provided the setting for "Tom Sawyer" and "Huckleberry Finn". He served an apprenticeship with a. In his autobiography, Twain writes of his early experiments with wearing white out-of-season: Samuel Langhorne Clemens (November 30, 1835 â€“ April 21, 1910), known by his pen name Mark Twain, was an American writer, humorist, entrepreneur, publisher, and lecturer. He was lauded as the "greatest humorist the United States has produced," and William Faulkner called him "the father of American literature". His novels include "The Adventures of Tom Sawyer" (1876) and its sequel, "Adventures of Huckleberry Finn" (1884), the latter of which has often been called the "Great American Novel". Twain was raised in Hannibal, Missouri, which later provided the
# Question: If Mark Twain had used his actual name in his literary works, what name would appear in his writings?
# #### ['Samuel Langhorne Clemens']
# '''
class StrategyQA_Adapter(Reasoning_Adapter):
    def __init__(self, prompt, config):
        self.config = config
        self.prompt = prompt
        super().__init__(
                config=config,
                prompt=prompt,
            )
        self.stop_criterion = stop_criterion
        self.get_accuracy = get_accuracy
        self.is_correct = is_correct
        self.qa_template = config["qa_template"]

    # COntext + Question goes into input
    def get_positive_ans(self, b):
        # Combine the context and question from the 'Input' field
        positive_ans = '\n'.join(b['Input']) + '\n#### '
        # Join answers from the 'answers' column
        if isinstance(b['answers'], list):
            positive_ans += ', '.join(b['answers'])  # Join list of answers with commas
        else:
            positive_ans += str(b['answers'])  # Handle single answer
        # Ensure the return value is always a list
        return [positive_ans]
        # Positive answer = Positive Input + Correct Answer
        # Negative Answer = Negative Input + Correct Answer
        # b is the instance of the batch. One prompt entry instance that we have
        # b is from the input data}
    def formulate_question(self, b):
        return b['Input']

    def extract_ground_truth(self, b):
        answers = b['answers']
        if isinstance(answers, list):
            return answers
        try:
            return literal_eval(answers)
        except (ValueError, SyntaxError):
            raise ValueError(f"Failed to parse 'answers': {answers}")
    
    

