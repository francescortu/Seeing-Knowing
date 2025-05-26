import torch
import os



def get_whoops_element_by_id(whoops_dataset, image_id):
    """
    Retrieve a whoops_dataset element by its image_id
    
    Args:
        whoops_dataset (list): List of whoops_dataset elements
        image_id (str): The image_id to search for
    
    Returns:
        dict: Matching whoops_dataset element or None if not found
    """
    for element in whoops_dataset:
        if element['image_id'] == image_id:
            return element
    return None
ollama_model_map = {
    "llava-7b": "llava:7b-v1.6-mistral-fp16",
    "llava-34b":"llava:34b",
    "llama-3.2-90b": "llama3.2-vision:90b",
    "gemma3:27b": "gemma3:27b",
}
def start_ollama(model_name: str = "llava-7b"):
    # run bash command to start ollama

    os.system("ollama serve &")
    os.system(f"ollama run {ollama_model_map[model_name]} &")



def cosine_similarity_matrix(A, abs: bool = False, chunk_size: int = 1024):
    if A.ndim > 2:
        A = A.squeeze()
    # Normalize the rows of the matrix A
    A_norm = A / A.norm(dim=1, keepdim=True)

    n = A_norm.size(0)
    device = A.device
    dtype = A.dtype

    # Initialize an empty tensor to store the cosine similarities
    cosine_sim_matrix = torch.zeros((n, n), device=device, dtype=dtype)

    # Compute cosine similarities in chunks
    for start_i in range(0, n, chunk_size):
        end_i = min(start_i + chunk_size, n)
        A_chunk = A_norm[start_i:end_i]  # Chunk of A_norm

        # Compute cosine similarity between A_chunk and all of A_norm
        sim_chunk = torch.matmul(A_chunk, A_norm.t())

        cosine_sim_matrix[start_i:end_i] = sim_chunk

    if abs:
        cosine_sim_matrix = torch.abs(cosine_sim_matrix)

    return cosine_sim_matrix