'''
학습기간 : 2025년 3월 19일 ~ 2025년 3월 19일
테디노트 LCEL & Runnable
출처 : https://www.youtube.com/watch?v=0X4Ks_nJUt8&list=PLIMb_GuNnFweShkx8-yorSjhwGKv9raR_&index=65
'''

#%%
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

#%%
# API 키 등록
current_directory = os.getcwd()
env_path = os.path.join(current_directory, '.env')

load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

#%% 
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{local} 지역에서 점심 맛집은 어디야?")
prompt

#%%
model = ChatOpenAI(temperature=0.1)

#%% LCEL : 질문 - 프롬프트 - 모델전달 - 출력
chain = prompt | model

# %% 할루시네이션이 심각함
chain.invoke({"local":"시청역"})

# %% 'Passthrough'

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

chain = {"local": RunnablePassthrough()} | prompt | model

chain.invoke("대한민국 서울 시청역").content
# %% 'RunnableParallel'

prompt1 = PromptTemplate.from_template("{local} 지역에서 점심 맛집은 어디야?")
prompt2 = PromptTemplate.from_template("{local} 지역에서 점심 먹고 난 뒤 갈만한 카페는 어디야?")

chain1 = {"local":RunnablePassthrough()} | prompt1 | model
chain2 = {"local":RunnablePassthrough()} | prompt2 | model

map_chain = RunnableParallel(a=chain1, b=chain2)
# %%

map_chain.invoke("대한민국 서울 중구 세종대로")
# %%

def combine_txt(text) :
    return text['a'].content + ' ' + text['b'].content

#%% 'RunnableLambda'

final_chain = map_chain | {"local" : RunnableLambda(combine_txt)} | PromptTemplate.from_template(
    "다음의 내용을 문맥에 맞게 이어 붙어줘. 이모티콘도 적절히 추가해주고:\n{local}"
) 
# %%

final_chain.invoke("대한민국 서울 중구 세종대로")
