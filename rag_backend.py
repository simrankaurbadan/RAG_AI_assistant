# Required Libraries
import openai
import chromadb
import json
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from llamaapi import LlamaAPI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv('OPEN_AI_API__KEY')
llama_key = os.getenv('LLAMA_API_KEY')

# Initialize OpenAI Embedding
embedding_function = OpenAIEmbeddingFunction(
    model_name='text-embedding-3-small',
    api_key=openai_api_key
)

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path="/db")
collection_name = 'chat_data' # change based on your database name
collection = client.get_collection(name=collection_name, embedding_function=embedding_function)

# Initialize LlamaAPI Client
llama = LlamaAPI(llama_key)

def generate_response(user_query):
    # Step 1: Retrieve from ChromaDB
    results = collection.query(
        query_texts=[user_query],
        n_results=5
    )

    retrieved_context = ""
    for idx, doc in enumerate(results['documents'][0]):
        retrieved_context += f"{doc}\nResponse: {results['metadatas'][0][idx]['output']}\n\n"

    prompt = f"""
    You are a polite and helpful customer support assistant designed to provide accurate and context-aware responses to user queries.
    Your name is VICTOR.

    User Query: {user_query}

    Relevant Past Conversations (Context for Reference):
    {retrieved_context}

    Instructions:
    1. First, determine if the current user query is a follow-up to the previous one. Use semantic similarity, continuity in topic, or pronouns like "that", "it", "those", etc., as hints.
    2. If it's a follow-up, retain the context from the previous conversation.
    3. If it’s a new topic or unrelated query, disregard prior context and base your answer only on the new context provided.
    4. Generate the most helpful and accurate response using the relevant context.
    5. Ensure the response is polite, concise, and user-friendly.
    6. Clearly list the specific documents or pieces of context (from 'Relevant Past Conversations') that were used to formulate the response.
    7. If the user query involves explanation-seeking keywords (such as "why", "explain", "reason", "how come"), provide a brief and clear explanation to support your response.
    8. If you need to address the user in any response, please use the name Simran.
    9. Return the **entire retrieved_context exactly as provided**, as a JSON array in the output.

    Output:
    Return the response strictly in the following JSON format:

    {{
        "is_followup": true or false,  // Indicate whether this was a follow-up query
        "response": "Your final response to the user query here. Be polite and provide the best possible solution.",
        "retrieved_context": {json.dumps(retrieved_context, indent=4)},
        "evaluation": {{
            "Explanation": "In detail, explain how you arrived at this response using the provided context.",
            "Relevance": "A one-line explanation of why the response is relevant to the user query.",
            "Relevance_score": "Return a relevance score percentage (0% to 100%) indicating how closely the response matches the user query based on semantic similarity."
        }}
    }}
    """

    # Step 3: Call Llama-2 for Response Generation
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = llama.run(api_request_json)

    # Step 4: Process and Return Response
    final_response_text = response.json()['choices'][0]['message']['content']

    try:
        final_response_json = json.loads(final_response_text)
    except json.JSONDecodeError:
        print("Invalid JSON returned by Llama API. Raw response below:")
        print(final_response_text)
        raise

    return final_response_json
