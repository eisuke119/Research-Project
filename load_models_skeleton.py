import torch
from transformers import AutoTokenizer, AutoModel

def get_embeddings(dna, model, tokenizer, mean_pool=True):
    inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
    hidden_states = model(inputs)[0]
    if mean_pool:
        embedding = torch.mean(hidden_states[0], dim=0)
    else:
        embedding = torch.max(hidden_states[0], dim=0)
    return embedding

def load_model(model_path):
    return AutoModel.from_pretrained(model_path, trust_remote_code=True)

def load_tokenizer(tokenizer_path):
    return  AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


if __name__ == "__main__":
    dna = "ATCG"
    model_path = "zhihan1996/DNABERT-S"
    tokenizer_path = "zhihan1996/DNABERT-S"

    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    embeddings = get_embeddings(dna, model, tokenizer)
    print(embeddings)
