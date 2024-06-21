import re
def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text)
    return text

text = "dit me may"

result = clean_spaces(text)

for i in result:
    print(i)