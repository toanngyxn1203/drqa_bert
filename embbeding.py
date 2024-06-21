import transformers as trans
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

def tokens_embedding(inputs ):
    encoded_input = tokenizer(inputs, 
                                return_tensors='pt') #return encode with tensor from torch 
    
    inputs_id = encoded_input['input_ids']
    attention_masks = encoded_input['attention_mask']
    with torch.no_grad():  # Notification for tensor don't need caculate gardient
        outputs = model(inputs_id, attention_mask=attention_masks)
    embeddings = outputs.last_hidden_state
    embedding = embeddings[0][2]
    return embedding


if __name__ == '__main__':
    text = ['Toàn']
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    out = tokenizer(text, 
                    return_tensors='pt')  # Return encode with tensor from tor
    input_ids = out['input_ids']
    attention_mask = out['attention_mask']
    with torch.no_grad():  # Không cần tính toán gradient trong quá trình inference
        outputs = model(input_ids, attention_mask=attention_mask)
    print("input id" , input_ids, "attensor mask",attention_mask)

    # outputs.last_hidden_state chứa các embedding vector cho mỗi token
    embeddings = outputs.last_hidden_state
    embedding = embeddings[0][2]
    print(embedding)
