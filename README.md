# GenAI Notes

A collection of my daily notes while learning **Generative AI** and **AI Engineering**.

This repository serves as my personal knowledge base where I document concepts, examples, and important takeaways as I progress through my learning journey.

## Progress
### Day 1

- Set up the Groq API
- Made my first LLM API call
- Learned the **User** role in prompts
- Understood the LLM request/response flow

### Day 2

- Learned about the **system** role
- Learned about **temperature** parameter
- Used it to control randomness and creativity of the LLM's response 


### Day 3

- Learned about **LLM tokens**

### Day 4

- Learned about **Pydentic and JSON**
- Learned about LLM response type

### Day 5

- Built a mini project - AI Resume Parser
- Parsed PDF files using PyPDF
- It parses the resume and prints Name, Mobile, Email, Experience, Skills, Education and Projects
- Future Scope : Score the resume according to the Job Description

### Day 6

- Learned about prompt engineering and how to give a clear, and specific prompt to an LLM
- 6 Steps to design a good prompt - Role, Task, Constraint, Output Format, OneShot/ZeroShot (Giving examples), Fallback

### Day 7

- Learned about ReAct technique
- Think -> Act -> Observe -> Answer
- Implemented a basic ReAct agent by following a tutorial (main.py)
- Found out that the tutorial used regex for parsing for tool calls
- So rebuilt the agent using JSON which is cleaner architectural design (agent.py)

Agent Flow

User Question --> LLM --> Response --> Tool Call --> Python Tool --> Observation --> LLM --> Response --> ... --> Final Answer
      
### Day 8

- Learned about the concept of **Prompt Chaining**
- Divided a complex task into multiple prompts to achieve desired result efficiently.
- Created a mini project

### Day 9

- Learned about Streaming, why is it used and how to implement streaming

### Day 10

- Learned about the fundamental concept of Retrieval and Knowledge Bases

### Day 11

- Learned about Embeddings
- Converted text into vectors 
- Checked cosine similarity between two vectors

### Day 12

- Implemented a tiny RAG Pipeline
- Converted knowledge base into embeddings
- Used cosine similarity to retrieve more semantically related context
- Generated LLM response with the retrieved context
''' indexing -> query processing -> retrieval -> generation '''

### Day 13 

- Implemented a RAG pipeline using Qdrant Vector Database
- Stored document embeddings as vectors in Qdrant
- Used cosine similarity to retrieve the most relevant chunks for a query
- Tested semantic search with queries that had no exact keyword match in the knowledge base
- Used the retrieved context to generate answers with Groq LLM
- Understood how Vector Databases enable semantic search in RAG systems

### Day 14

- Learned about HNSW indexing algorithm (Hierarchical Navigable Small Words)
- Learned about qdrant filter based search
- Implemented filter based search in a RAG system
- Generated LLM response with the retrieved context and Groq API