import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings


# COURSE TOPICS
QUERY = False
STEB_BY_STEP = True

# load API KEY
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

file = 'data/OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)


# MAKING A SIMPLE QUERY
if QUERY:
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])

    query ="Please list all your shirts with sun protection \
    in a table in markdown and summarize each one."
    response = index.query(query)
    print(response)

# STEB BY STEP
if STEB_BY_STEP:
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    # embed = embeddings.embed_query("Hi my name is Harrison")
    # print(embed)

    # creating a vector database in memory
    db = DocArrayInMemorySearch.from_documents(
        docs,
        embeddings
    )

    # making a query on vector database
    query = "Please suggest a shirt with sunblocking"
    docs = db.similarity_search(query)

    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0.0)

    qdocs = "".join([docs[i].page_content for i in range(len(docs))])
    response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
    shirts with sun protection in a table in markdown and summarize each one.")

    # Creating a Retrieval Chain
    qa_stuff = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    query =  "Please list all your shirts with sun protection in a table \
    in markdown and summarize each one."
    response = qa_stuff.run(query)
    print(response)
