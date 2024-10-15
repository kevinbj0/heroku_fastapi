import os
import openai
import asyncio
import pandas as pd
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from llama_index.llms.openai import OpenAI
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)

# 환경 변수에서 API 키 읽기
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 상대 경로로 CSV 파일 읽기
df = pd.read_csv('data/모니카데이터_전자제품.csv')
print(df.head())

# LLM 초기화
ft_llm = OpenAI(model="gpt-4o")

# Prompt 및 QueryPipeline 설정
instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

system_prompt_str = """
    "당신은 대유백화점의 AI 상담원으로 전자제품을 추천하거나 팔고 있습니다."
    "전자제품에 대한 추천은 df의 특정 컬럼을 참고해서 처리해야 합니다."
"""
system_prompt = PromptTemplate(system_prompt_str)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "All response should be generated in Korean.\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

qp = QP(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": ft_llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": ft_llm,
    },
    verbose=False,
)
qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
    ]
)
qp.add_link("response_synthesis_prompt", "llm2")

# FastAPI 앱 초기화
app = FastAPI()

@app.post('/')
async def post_example(request: Request):
    request_data = await request.json()
    record_id = request_data.get("recordId")
    request_message = str(request_data.get("Message"))

    # 사용자 메시지 로그
    print(f'사용자 ({record_id}): {request_message}') 

    user_msg = ChatMessage(role="user", content=request_message)
    chat_history_str = f"User: {request_message}"

    # AI 응답
    response = await asyncio.to_thread(qp.run, query_str=chat_history_str)
    response_message = response.message.content

    # 응답 메시지 로그
    print(f'답변 ({record_id}): {response_message}')

    return JSONResponse(content={'message': response_message})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
