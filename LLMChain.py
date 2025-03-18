'''
랭체인 입문부터 응용까지
출처 : https://wikidocs.net/231156
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
# model 
llm = ChatOpenAI(model="gpt-4o-mini")

# chain 실행
llm.invoke("지구의 자전 주기는?")


#%% 프롬프트 적용

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("너는 천문학자야. 다음 질문에 답변해봐. <질문>: {input}")
prompt

#%% 체인 연결

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# chain 연결 (LCEL)
chain = prompt | llm

# chain 호출
chain.invoke({"input": "지구의 자전 주기는?"})
# %%

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# prompt + model + output parser
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# LCEL chaining
chain = prompt | llm | output_parser

# chain 호출
chain.invoke({"input": "지구의 자전 주기는?"})

# %% 순차적 체인 연결

prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
prompt2 = ChatPromptTemplate.from_template(
    "explain {english_word} using oxford dictionary to me in Korean."
)

llm = ChatOpenAI(model="gpt-4o-mini")

chain1 = prompt1 | llm | StrOutputParser()

chain1.invoke({"korean_word":"미래"})
# %%

chain2 = (
    {"english_word": chain1}
    | prompt2
    | llm
    | StrOutputParser()
)

chain2.invoke({"korean_word":"미래"})
# %%
