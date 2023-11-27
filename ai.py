import time
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

print("Starting AI...")

# Prompt
template = """[INST] <<SYS>> Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. <</SYS>>
{context}
Question: {question}
Helpful Answer:[/INST]"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=40,
    n_batch=512,
    n_ctx=8192,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=False,
)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

qachain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,
    chain_type="stuff",
    )

print("Type 'exit' to leave\n")
while True:
    question = input("> ")

    if question == "exit":
        break
    
    if question.strip() == "":
        continue

    begin = time.time()
    res = qachain({"query": question})
    end = time.time()
    print()
    if len(res['source_documents']) != 0:
        print("\nSources:")
        for document in res['source_documents']:
            print(f"- {document.metadata['source']} (page {document.metadata['page']}/{document.metadata['total_pages']})")
    print(f"\n(Answered in {round(end-begin, 2)}s)\n")
