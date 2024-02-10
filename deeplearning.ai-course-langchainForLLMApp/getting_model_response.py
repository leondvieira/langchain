import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# load API KEY
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


# Defining the model
LLM_MODEL = "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0.0, model=LLM_MODEL)


style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

# CHECK PROMPT TEMPLATE VARIABLES
# print(prompt_template.messages[0].prompt.input_variables)

customer_messages = prompt_template.format_messages(
    style=style,
    text=customer_email
)
# CHECK PROMPT TEMPLATE WITH MESSAGE
# print("\n\n" + str(customer_messages[0]) +"\n\n")

# GET MODEL RESPONSE
customer_response = chat(customer_messages)
print(customer_response.content)