import os
import openai
import warnings

from dotenv import load_dotenv, find_dotenv

from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


warnings.filterwarnings("ignore")

# load API KEY
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

tools = load_tools(["llm-math","wikipedia"], llm=llm)

agent= initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True
)

agent("What is the 25% of 300?")
