import re

def clean_token(token: str) -> str:
    token = token.lower()
    token = re.sub(r'[^a-z0-9]', '', token)
    return token
