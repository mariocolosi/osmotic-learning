# v0.1.0

import torch
import numpy as np
import random


def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cosine_similarity(vec_a, vec_b):
    return cosine_similarity_exponential(vec_a, vec_b)

def cosine_similarity_exponential(vec_a, vec_b, beta=2):
    """
    Calculates the cosine similarity between two vectors, with an exponential penalty.
    """
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()
    
    cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    similarity = np.sign(cosine_sim) * (np.abs(cosine_sim) ** beta)
    
    return similarity

def cosine_similarity_penalized(vec_a, vec_b, alpha=0.5):
    """
    Calculates the cosine similarity between two vectors, with a penalization factor.
    """
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()
    
    cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    euclidean_dist = np.linalg.norm(vec_a - vec_b)
    
    similarity = cosine_sim - alpha * euclidean_dist
    
    return similarity