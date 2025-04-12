from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag_backend import generate_response
from generate_logs import log_to_excel

import uvicorn

app = FastAPI()

templates = Jinja2Templates(directory="Templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Home Page
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("Home-Page.html", {"request": request})

# Chat Page (GET)
@app.get("/generate_response", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("CHATBOT.html", {"request": request})

# Chat Page (POST)
@app.post("/generate_response", response_class=HTMLResponse)
async def post_chat(request: Request, user_query: str = Form(...)):
    print("User Query:", user_query)

    response = generate_response(user_query)

    response_text = response['response']
    retrieved_context = response['retrieved_context']
    relevance_score = response['evaluation']['Relevance_score']
    explanation = (
        str(response['evaluation']['Explanation']) +
        ' Also the Past retrieved context from vector DB are : ' +
        str(retrieved_context)
    )

    # Log the interaction
    log_to_excel(user_query, response_text, retrieved_context, relevance_score)

    return JSONResponse(content={
        "response": response_text,
        "explainability_reason": explanation,
        "similarity_score": relevance_score
    })

# Entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
