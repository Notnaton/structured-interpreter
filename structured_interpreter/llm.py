from typing import List
from os import environ

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatLiteLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def main():
    environ['OPENAI_API_KEY'] = "key"
    model = ChatLiteLLM(
        model_name="local",
        api_base="http://localhost:1234/v1",
        custom_llm_provider="openai",
        temperature=0,
        streaming=True,
        #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # Define your desired data structure.
    class Message(BaseModel):
        title: str = Field(description="Three word title")
        plan: List[str] = Field(description="create a complete step by step plan, Step 1: ..., Step 2: ..., Step 10: ...")
        code: List[str] = Field(description="fill in code from each step of the plan")

    user_message = "Write a hello world program using python"

    parser = PydanticOutputParser(pydantic_object=Message)

    prompt = PromptTemplate(
        template="Answer the user message with as many steps as needed.\n{format_instructions}\n{user_message}\n",
        input_variables=["user_message"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    #chain.invoke()

    handler = StreamingStdOutCallbackHandler()

    output = chain.invoke({"user_message": user_message}, {"callbacks":[handler]})
    print(output)
