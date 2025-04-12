# RAG_AI_assistant
Developed an AI-powered customer support assistant that enhances responses using a foundational LLM and retrieval-augmented generation (RAG). Uses chromab for creating vector database, Openai and LLama as embedding and response generation models.

# Objective
To develop an AI-powered customer support assistant that enhances responses using a foundational LLM and retrieval-augmented generation (RAG).

# Database
We will use a public customer support dataset:
Link to download - https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k
This dataset contains customer queries and support responses. You can use this for retrieval-based response generation.

# Solution built and files explanation
This is a RAG based AI assistant. It uses FASTAPI for creating the /generate_response API and also for running the frontend UI. 


chromadb_vector_creation.py - This has the code for building a vector database. You would need OPENAI_API_KEY since I have used OPENAI for creating the embeddings since its fast. 
The user will have to make chnges to how much datapoints they want to embedd and also change the collection name as per their choice. The code will create a persistent database meaning save the database in your local by creating a folder db_sample in your working directory.

generate_logs.py - This is a function to save the logs to an excel file for later reference. 

rag_backend.py - this will use your created database collection to get the top 5 matches based on user query and then pass those retrieved context to the LLM for response generation. This uses OPENAI_API_KEY and LLAMA_API_KEY for running the embedding as well as calling the Llama models to generate rhe response.

main.py - This file has the main FASTAPI application code which is combining everything and working to create a frontend application where the user can go and chat. 
