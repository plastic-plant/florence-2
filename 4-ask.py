# RAG with NoMAD and GEMMA-2. Answers questions on documents matching in Chroma database.
# Requires Ollama, nomic-embed-text and gemma2 models. Pull in these models with Ollama
# 
#
#    install https://ollama.com/download  # Runs at localhost:11434
#    ollama pull nomic-embed-text         # Size: 275 MB
#    ollama pull gemma2                   # Size: 5.5 GB | See: https://ollama.com/library
#
#    pip3 install chromadb langchain langchain_community getch
#
#    python3 4-ask.py

# Output:
#
#   Answering questions on documents matching in Chroma database with GEMMA-2.
#   Ask a question, press [CTRL+C] or [ENTER] without a question to quit.
#
#   Ask:     Who established the Royal Navy?
#   Answer:  Henry VIII established the Royal Navy.
#
#   Ask:     Who was married to Adelaide? What source?
#   Answer:  William IV was married to Princess Adelaide of Saxe-Meiningen.
#            SOURCE: Document 1

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from platform import system
if system() == 'Windows':
    from msvcrt import getch
else:
    from getch import getch

embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)
chroma = Chroma(persist_directory="chroma", collection_name="florence_collection", embedding_function=embeddings)
retriever = chroma.as_retriever(search_type="similarity", search_kwargs= {"k": 5})
ollama = ChatOllama(model='gemma2', keep_alive="10m", max_tokens=8192, temperature=0)
prompt = """<bos><start_of_turn>user\nAnswer the question short with following CONTEXT only. If there's no context given, explain that. If the context doen't contain the answer, tell that.\n\nCONTEXT: {context}\n\nQUESTION: {question}\n\n<end_of_turn><start_of_turn>model\n\nANSWER:"""
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(prompt)
    | ollama
)

def answer(question):
    print("\nAnswer: ", end="")
    for chunk in chain.stream(question):
        print(chunk.content, end="", flush=True)

print("Answering questions on documents matching in Chroma database with GEMMA-2.\nAsk a question, press [CTRL+C] or [ENTER] without a question to quit.")
try:
    while True:
        question = input("\nAsk: ")
        if question == '': break
        answer(question)

except KeyboardInterrupt:
    pass

finally:
    print("\n")
