import os
import signal
import sys
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")



def signal_handler(sig, frame):
    print("\n Thanks for using the system")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
# generating a part of total system that we are creating that takes in query that takes in query and returns part relevant to the query

def get_relevant_context_from_db(query):
    context=""
    embeddings_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':'cpu'})
    vector_db=Chroma(persist_directory="./chroma_db_nccn", embedding_function=embeddings_function)
    search_results=vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content+"\n"
    return context

def generate_rag_prompt(query, context):
    escaped=context.replace("'","").replace('"',"").replace("\n", " ")
    prompt= ("""
          You are a helpful and informative bot that answers questions using text from the reference context included below. \
             Be sure to respond in a complete sentence being comprehensive, including all relevant background information. \
             However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
             strike a friendly and conversational tone. You should also be able to relate abbreviated query to its actual meaning. \
             You should also be able to check the appendices in the document provided to link up where the context is present. \
             If the context is irrelevant to the answer, you may ignore it 
            QUESTION:'{query}'
             CONTEXT='{context}'

             ANSWER:
            """).format(query=query, context=context)
    return prompt

def generate_answer(prompt):
    # load environment variable form .env file
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    answer=model.generate_content(prompt)
    return answer.text

welcome_text=generate_answer("Can you quikcly introduce yourself?")

print(welcome_text)

while True:
    print("-----------------------------------------------------------------------------------------")
    print("What would you like to ask?")
    query=input("Query:")
    context=get_relevant_context_from_db(query=query)
    prompt=generate_rag_prompt(query=query, context=context)
    answer=generate_answer(prompt)
    print(answer)
    
