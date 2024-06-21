import transformers as trans
from transformers import BertTokenizer
import torch
from embbeding import tokens_embedding
#split squences from 
def split_sequens( tokens = ""):
    token = tokens.split(".")
    token = token[0].split(" ")
    return  token

def return_embeding(tokens = ""):
    re = split_sequens(tokens)
    embeddings  = []
    for i in re:
       embeddings.append(tokens_embedding(i))
    return embeddings
    


if __name__ == '__main__':
    re = "Toan dep trai vcllllll."
    re = return_embeding(tokens= re)
    for i in re:
        print(i)