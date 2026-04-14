import streamlit as st
import numpy as np
import keras
import pickle
import warnings
warnings.filterwarnings('ignore')

# 🚀 DEEP PATCH: Recursively remove quantization_config from Keras config
from keras.src.saving import serialization_lib

def strip_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop('quantization_config', None)
        for key, value in obj.items():
            strip_quantization_config(value)
    elif isinstance(obj, list):
        for item in obj:
            strip_quantization_config(item)
    return obj

original_deserialize_keras_object = serialization_lib.deserialize_keras_object

def patched_deserialize_keras_object(config, *args, **kwargs):
    if isinstance(config, dict):
        config = strip_quantization_config(config)
    return original_deserialize_keras_object(config, *args, **kwargs)

serialization_lib.deserialize_keras_object = patched_deserialize_keras_object

from keras.saving import load_model
from keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(page_title="LSTM Generator", page_icon="📝")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; }
    .main-header { text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 15px; }
    .generated-text { background: white; padding: 20px; border-radius: 10px; color: #333; font-size: 1.1rem; border-left: 5px solid #2a5298; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all():
    try:
        model = load_model('LSTM.keras')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('max_sequence_len.pkl', 'rb') as f:
            max_sequence_len = pickle.load(f)
        return model, tokenizer, max_sequence_len
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

def main():
    st.markdown('<div class="main-header"><h1>📝 LSTM Text Generator</h1></div>', unsafe_allow_html=True)
    
    model, tokenizer, max_sequence_len = load_all()
    if not model: return

    seed = st.text_area("Seed Text", "The future of AI is")
    num = st.slider("Words", 10, 100, 50)
    temp = st.slider("Creativity", 0.1, 2.0, 1.0)
    
    if st.button("Generate"):
        with st.spinner("Writing..."):
            text = seed
            for _ in range(num):
                tokens = tokenizer.texts_to_sequences([text])[0]
                tokens = pad_sequences([tokens], maxlen=max_sequence_len-1, padding='pre')
                preds = model.predict(tokens, verbose=0)[0]
                
                # Temperature sampling
                preds = np.log(preds + 1e-7) / temp
                exp_preds = np.exp(preds)
                preds = exp_preds / np.sum(exp_preds)
                idx = np.random.multinomial(1, preds, 1).argmax()
                
                word = tokenizer.index_word.get(idx, "")
                if not word: break
                text += " " + word
            
            st.markdown(f'<div class="generated-text">{text}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()