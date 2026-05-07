from langchain_aws import ChatBedrockConverse

from dotenv import load_dotenv

load_dotenv()

model = ChatBedrockConverse(
    model="amazon.nova-pro-v1:0",
)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to marathi."),
    ("human", "I love programming."),
]

print(model.invoke(messages))