import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle

def save_tokenizer_and_params(tokenizer, max_sequence_len):
    """Save tokenizer and parameters for later use"""
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('max_sequence_len.pkl', 'wb') as f:
        pickle.dump(max_sequence_len, f)

def load_tokenizer_and_params():
    """Load tokenizer and parameters"""
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('max_sequence_len.pkl', 'rb') as f:
        max_sequence_len = pickle.load(f)
    return tokenizer, max_sequence_len

def sample_with_temperature(preds, temperature=1.0):
    """Apply temperature sampling to predictions with numerical stability"""
    preds = np.asarray(preds).astype('float64')
    # Use a small epsilon and the softmax trick (subtract max) for stability
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds - np.max(preds))
    preds = exp_preds / np.sum(exp_preds)
    # Re-normalize to ensure the sum is exactly 1.0 for np.random.choice
    preds = preds / np.sum(preds)
    return np.random.choice(len(preds), p=preds)