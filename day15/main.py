from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text = """
Artificial Intelligence is changing the way we build software. Large Language Models can understand and generate human-like text. They are used in chatbots, coding assistants, search systems, and many other applications.

Retrieval Augmented Generation, commonly known as RAG, allows an LLM to retrieve relevant information from external sources before generating an answer. This helps reduce hallucinations and allows the model to work with private or frequently changing data.

Chunking is an important part of a RAG pipeline. Large documents are divided into smaller chunks before they are converted into embeddings. These embeddings are then stored in a vector database and retrieved when a user asks a question.

There are different strategies for chunking documents. Character-based splitting divides text based on a specified separator and chunk size. Recursive character splitting tries multiple separators, such as paragraphs, new lines, sentences, and spaces, to create more meaningful chunks.

Choosing the right chunk size is important. If chunks are too small, important context may be lost. If chunks are too large, retrieval may return unnecessary information and consume more tokens. The ideal configuration depends on the type of documents and the application's retrieval requirements.
"""

# ---------------FIXED SIZE CHUNKING---------------

fixed = CharacterTextSplitter(
    separator = "", 
    chunk_size = 100, 
    chunk_overlap = 0
)

print("############ fixed size ############ \n")

for i, chunk in enumerate(fixed.split_text(text)):
    print(f"Chunk {i+1}: {chunk}\n")
    print("---------------")

# ---------------PARAGRAPH CHUNKING---------------

paragraph = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 100,
    chunk_overlap = 0
)

print("############ paragraph chunking ############ \n")

for i, chunk in enumerate(paragraph.split_text(text)):
    print(f"chunk {i+1}: {chunk}\n")
    print("---------------")

# ---------------RECURSIVE CHUNKING---------------

recursive = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

print("############ recursive ############ \n")


for i, chunk in enumerate(recursive.split_text(text)):
    print(f"chunk {i+1}: {chunk}\n")
    print("---------------")

