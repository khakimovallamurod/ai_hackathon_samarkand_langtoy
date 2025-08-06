from dotenv import load_dotenv
import os

load_dotenv()
def get_token():
    token = os.getenv('TOKEN')
    if token is None:
        return "Not found TOKEN"
    return token

def get_chat_id():
    chat_id = os.getenv('CHAT_ID')

    if chat_id is None :
        return "Not found CHAT ID"
    return chat_id
def get_open_ai_key():
    api_key = os.getenv('OPENAI_API_KEY')

    if api_key is None :
        return "Not found CHAT ID"
    return api_key