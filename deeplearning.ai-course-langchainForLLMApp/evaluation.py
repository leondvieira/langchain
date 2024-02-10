import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores.docarray import DocArrayInMemorySearch
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain


# COURSE TOPICS
MANUAL_EVAL = False
ASSISTED_EVAL = True


# load API KEY
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


# Create our Q&A Application
file = 'data/OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>>"
    }
)

# Testing documents
# print(data[10])
# print(data[11])

# Hard-coded question examples
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# LLM-Generated question Examples
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

examples += new_examples

# Manual Evaluation
if MANUAL_EVAL:
    import langchain
    langchain.debug=True
    qa.run(examples[0]["query"])

# LLM ASSISTED Evaluation
if ASSISTED_EVAL:
    predictions = qa.apply(examples)
    eval_chain = QAEvalChain.from_llm(llm=llm)

    graded_outputs = eval_chain.evaluate(examples, predictions)
    for i, eg in enumerate(examples):
        print("\n"*3)
        print("____________________")
        print(f"Example {i}:")
        print("Question: " + predictions[i]['query'])
        print("Real Answer: " + predictions[i]['answer'])
        print("Predicted Answer: " + predictions[i]['result'])
        print("Predicted Grade: " + graded_outputs[i]['text'])
        print("\n"*3)
        print("____________________")
