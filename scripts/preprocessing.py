import re

def clean_text(token):
    token = token.lower()
    token = re.sub(r'[^a-z0-9]', '', token)
    return token