import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory
)

# load API KEY
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


# Defining the model
LLM_MODEL = "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0.0, model=LLM_MODEL)

# COURSE TOPICS
SIMPLE_CONVERSATION = False
INPUTTING_CONTEXT = False
BUFFER_WINDOW_MEMORY = False
TOKEN_BUFFER_MEMORY = False
SUMMARY_MEMORY = True


#########################################################################
# A SIMPLE CONVERSATION WITH MEMORY

if SIMPLE_CONVERSATION:
    # A SIMPLE CONVERSATION
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=chat,
        memory=memory,
        verbose=True
    )

    print("\n"*2)
    print("-"*30)
    # INPUT A MESSAGE
    print(conversation.predict(input="Hi, my name is Leonardo"))
    print(conversation.predict(input="What is 1+1?"))
    print(conversation.predict(input="What is my name?"))

    print("\n"*2)
    print("-"*30)
    # SHOWS CONVERSATION HISTORY
    print(memory.buffer)

    # GET CONVERSATION MEMORY VARIABLES
    print(memory.load_memory_variables({}))

#########################################################################
# INPUTTING CONTEXT MESSAGES TO CONVERSATION

if INPUTTING_CONTEXT:
    memory.save_context(
        {"input": "Hi"},
        {"output": "What's up"}
    )

    print(memory.buffer)

#########################################################################
# CONVERSATION BUFFER WINDOW MEMORY

if BUFFER_WINDOW_MEMORY:
    # "k" is the quantity of "last messages" that we can choose to keep on the memory buffer
    # k=1 means that we keep only the last message from AI and the Client
    memory = ConversationBufferWindowMemory(k=1)
    memory.save_context(
        {"input": "Hi"},
        {"output": "What's up"}
    )
    memory.save_context(
        {"input": "Not much, just hanging"},
        {"output": "Cool"}
    )

    # print(memory.load_memory_variables({}))

    memory = ConversationBufferWindowMemory(k=1)
    conversation = ConversationChain(
        llm=chat,
        memory = memory,
        verbose=False
    )
    print(conversation.predict(input="Hi, my name is Andrew"))
    print(conversation.predict(input="What is 1+1?"))
    print(conversation.predict(input="What is my name?"))

#########################################################################
# LIMIT MAX TOKEN CONVERSATION

if TOKEN_BUFFER_MEMORY:
    memory = ConversationTokenBufferMemory(llm=chat, max_token_limit=20)
    memory.save_context(
        {"input": "AI is what?!"},
        {"output": "Amazing!"}
    )
    memory.save_context(
        {"input": "Backpropagation is what?"},
        {"output": "Beautiful!"}
    )
    memory.save_context(
        {"input": "Chatbots are what?"},
        {"output": "Charming!"}
    )

    print(memory.load_memory_variables({}))

#########################################################################
# CONVERSATION SUMMARY MEMORY

if SUMMARY_MEMORY:
    schedule = "There is a meeting at 8am with your product team. \
    You will need your powerpoint presentation prepared. \
    9am-12pm have time to work on your LangChain \
    project which will go quickly because Langchain is such a powerful tool. \
    At Noon, lunch at the italian resturant with a customer who is driving \
    from over an hour away to meet you to understand the latest in AI. \
    Be sure to bring your laptop to show the latest LLM demo."

    memory = ConversationSummaryBufferMemory(llm=chat, max_token_limit=100)
    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"},
                        {"output": "Cool"})
    memory.save_context({"input": "What is on the schedule today?"},
                        {"output": f"{schedule}"})

    print(memory.load_memory_variables({}))
