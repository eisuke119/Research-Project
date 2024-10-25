import torch
from transformers import AutoTokenizer, AutoModel

def get_embeddings(dna:str, model:AutoModel, tokenizer:AutoTokenizer, mean_pool:bool=True) -> torch.Tensor:
    """Get embeddings for a DNA sequence given a model and tokenizer

    Args:
        dna (str): The DNA Sequence
        model (AutoModel): The loaded huggingface model
        tokenizer (AutoTokenizer): The loaded huggingface tokenizer
        mean_pool (bool, optional): Pooling type. Defaults to True.

    Returns:
        torch.Tensor: The embeddings for the DNA sequence
    """
    inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"]
    hidden_states = model(inputs)[0]
    if mean_pool:
        embedding = torch.mean(hidden_states[0], dim=0)
    else:
        embedding = torch.max(hidden_states[0], dim=0)
    return embedding

def load_model_and_tokenizer(model_path:str, tokenizer_path) -> AutoModel:
    """Loads the model and tokenizer from huggingface
    Args:
        model_path (str): The model path
        tokenizer_path (str): The tokenizer path

    Returns:
        AutoModel: The loaded model
        AutoTokenizer: The loaded tokenizer
    """
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

if __name__ == "__main__":
    dna = "ATCG"
    model_path = "zhihan1996/DNABERT-S"
    tokenizer_path = "zhihan1996/DNABERT-S"

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    embeddings = get_embeddings(dna, model, tokenizer)
