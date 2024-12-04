# Donut
Run LangchainSave.py
Initial idea is book reviews
Not as relevant to work as expected, shifting to Pathfner 2e PDF RAG 
Problem Number #1: 
Chunking the PDFS in a way that is relevant and useful but can be done on my laptop
https://www.anthropic.com/news/contextual-retrieval
Solutions:
Langchain build in framework for chunking and pipelining into LLM/SLM 
https://python.langchain.com/docs/concepts/rag/
https://python.langchain.com/docs/tutorials/rag/
Note need to install langchain-community (pip)
Problem Number #2:
Function while utilizing distilGPT-2 but results are poor
Need to use something with bigger context
Qwen 2 
Does not function due to sysmlink error
Solution: Run as Admin
