# ==========================================================
# BASIC RAG USING LANGCHAIN 1.x + OLLAMA + CHROMA
# ==========================================================

# --------------------------
# DOCUMENT LOADING
# --------------------------

from langchain_community.document_loaders.text import TextLoader

# --------------------------
# TEXT SPLITTING
# --------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------
# EMBEDDINGS
# --------------------------

from langchain_ollama import OllamaEmbeddings

# --------------------------
# VECTOR DATABASE
# --------------------------

from langchain_chroma import Chroma

# --------------------------
# LLM
# --------------------------

from langchain_ollama import ChatOllama

# ==========================================================
# STEP 1 : LOAD DOCUMENT
# ==========================================================

print("\nLoading document...\n")

loader = TextLoader(
    "docs/company_policy.txt",
    encoding="utf-8"
)

documents = loader.load()

print("Document Loaded Successfully")
print(f"Number of documents: {len(documents)}")

# ==========================================================
# STEP 2 : SPLIT DOCUMENT INTO CHUNKS
# ==========================================================

print("\nSplitting document into chunks...\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks Created: {len(chunks)}")

for index, chunk in enumerate(chunks):
    print(f"\nChunk {index + 1}")
    print(chunk.page_content)
    print("-" * 50)

# ==========================================================
# STEP 3 : CREATE EMBEDDING MODEL
# ==========================================================

print("\nLoading Embedding Model...\n")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding Model Loaded")

# ==========================================================
# STEP 4 : CREATE VECTOR DATABASE
# ==========================================================

print("\nCreating Chroma Vector Store...\n")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector Store Created Successfully")

# ==========================================================
# STEP 5 : CREATE RETRIEVER
# ==========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# ==========================================================
# STEP 6 : LOAD LLM
# ==========================================================

print("\nLoading LLM...\n")

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0
)

print("LLM Loaded Successfully")

# ==========================================================
# STEP 7 : QUESTION ANSWERING LOOP
# ==========================================================

while True:

    question = input(
        "\nAsk a question (type 'exit' to quit): "
    )

    if question.lower() == "exit":
        break

    # ------------------------------------------------------
    # RETRIEVE RELEVANT CHUNKS
    # ------------------------------------------------------

    retrieved_docs = retriever.invoke(question)

    print("\n===== RETRIEVED CHUNKS =====")

    for index, doc in enumerate(retrieved_docs):
        print(f"\nChunk {index + 1}")
        print(doc.page_content)

    # ------------------------------------------------------
    # BUILD CONTEXT
    # ------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # ------------------------------------------------------
    # CREATE PROMPT
    # ------------------------------------------------------

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
reply with:

"I could not find that information in the document."

Context:
{context}

Question:
{question}
"""

    # ------------------------------------------------------
    # CALL LLM
    # ------------------------------------------------------

    response = llm.invoke(prompt)

    # ------------------------------------------------------
    # PRINT ANSWER
    # ------------------------------------------------------

    print("\n===== ANSWER =====\n")
    print(response.content)