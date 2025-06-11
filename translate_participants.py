import re

SOURCE_PATH = '/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/preparticipants.csv'
OUT_PATH = '/Users/owenflynn/research/MENTAL2/TD-BRAIN-sample/participants.csv'

def replace_whitespace(text):
    return re.sub(r'[\t\x0b\x0c\r\f ]+', ',', text)

with open(SOURCE_PATH, 'r') as src:
    content = src.read()

modified_content = replace_whitespace(content)

with open(OUT_PATH, 'w') as out:
    out.write(modified_content)
