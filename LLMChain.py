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
llm.invoke("지구의 자전 주기는?, 간결하게 대답")


