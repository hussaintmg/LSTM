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
    """Apply temperature sampling to predictions"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)