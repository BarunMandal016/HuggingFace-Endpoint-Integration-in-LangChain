from langchain_huggingface import HuggingFaceEndpoint
from decouple import config
HUGGINGFACEHUB_API_TOKEN = config('HUGGINGFACEHUB_API_TOKEN')
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

question = "give me 500 word on gautam buddha "

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
     callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
    max_new_tokens=1000,
    stop_sequences=["</s>"],
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
llm_chain = prompt | llm
print(llm_chain.invoke({"question": question}))