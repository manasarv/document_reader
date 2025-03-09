# %%
import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import os

# %%
##Lets read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

# %%
doc = read_doc('documents/')
doc

# %%
len(doc)

# %%
##Divide the docws into chunks

def chunk_data(docs, chunk_size=800,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return docs

# %%
documents = chunk_data(docs=doc)
documents

# %%
len(documents)

# %%
from langchain_openai import OpenAIEmbeddings


# %%
embeddings= OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
embeddings

# %%
vectors=embeddings.embed_query("How are you?")
len(vectors)

# %%
#Vector Search in Pinecone
#from pinecone import Pinecone

#pc = Pinecone(api_key="pcsk_3E4CXd_KGS72QDj7UgiXwY45aDxGYCPoBv3BFbgoS2JYrYvavV6krtsYqjzhtPzKcfJa8n")
#index_name = pc.Index("langchainvector")

# %%

#os.environ["PINECONE_API_KEY"] = "pcsk_3E4CXd_KGS72QDj7UgiXwY45aDxGYCPoBv3BFbgoS2JYrYvavV6krtsYqjzhtPzKcfJa8n"

# %%
#import os
##from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone instance
#pc = Pinecone(api_key="pcsk_3E4CXd_KGS72QDj7UgiXwY45aDxGYCPoBv3BFbgoS2JYrYvavV6krtsYqjzhtPzKcfJa8n")

# List available indexes (optional, to verify connection)
#print(pc.list_indexes().names())

# %%
from langchain_pinecone import PineconeVectorStore
import pinecone

# %%
#pc = Pinecone(api_key="pcsk_3E4CXd_KGS72QDj7UgiXwY45aDxGYCPoBv3BFbgoS2JYrYvavV6krtsYqjzhtPzKcfJa8n")

# %%
vectorstore = PineconeVectorStore(
    index_name="langchainvector",
    pinecone_api_key="pcsk_3E4CXd_KGS72QDj7UgiXwY45aDxGYCPoBv3BFbgoS2JYrYvavV6krtsYqjzhtPzKcfJa8n",
    embedding=embeddings
)


# %%
vectorstore.add_documents(doc)

# %%
def retrieve_query(query, k=2):
    matching_results =  vectorstore.similarity_search(query, k =k)
    return matching_results

# %%
#from langchain.chains.question_answering import load_qa_chain
#from langchain import OpenAI

# %%
def retrieve_answers(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response= chain.run(input_documents=doc_search, question=query)
    return response

# %%
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chain = load_qa_chain(llm, chain_type="stuff")  

# %%
our_query="what is hapenning with Make AI in India and Make AI work for India"
answer = retrieve_answers(our_query)
print(answer) 

