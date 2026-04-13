import streamlit as st
import numpy as np
import keras
import pickle
import warnings
warnings.filterwarnings('ignore')

# 🔥 BRUTE FORCE FIX: Monkeypatch Keras Embedding layer to ignore quantization_config
import keras.layers
original_embedding_init = keras.layers.Embedding.__init__
def patched_embedding_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_embedding_init(self, *args, **kwargs)
keras.layers.Embedding.__init__ = patched_embedding_init

from keras.saving import load_model
from keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="LSTM Text Generator",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main-header { text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 20px; margin-bottom: 2rem; backdrop-filter: blur(10px); }
    .generated-text { background: rgba(255,255,255,0.95); padding: 2rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-top: 2rem; font-size: 1.2rem; line-height: 1.6; color: #2c3e50; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.8rem 2rem; font-weight: bold; border-radius: 10px; }
    .metric-card { background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 10px; text-align: center; backdrop-filter: blur(5px); color: white; }
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
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def generate_text(model, tokenizer, seed_text, num_words, max_sequence_len, temperature=0.9):
    generated_text = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        
        # Sampling
        preds = np.asarray(predicted).astype('float64')
        preds = np.log(preds + 1e-7) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        idx = np.random.multinomial(1, preds, 1).argmax()
        
        word = tokenizer.index_word.get(idx, "")
        if not word: break
        generated_text += " " + word
    return generated_text

def main():
    st.markdown('<div class="main-header"><h1 style="color: white;">📝 LSTM Text Generator</h1><p style="color: white;">By Deep Learning</p></div>', unsafe_allow_html=True)
    
    model, tokenizer, max_sequence_len = load_all()
    if not model: return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### ✍️ Input")
        seed = st.text_area("Start your story...", height=100)
        num = st.number_input("Words", 1, 200, 50)
        temp = st.slider("Creativity", 0.1, 2.0, 0.9)
        if st.button("🚀 Generate", use_container_width=True):
            with st.spinner("Thinking..."):
                result = generate_text(model, tokenizer, seed, num, max_sequence_len, temp)
                st.markdown(f'<div class="generated-text">{result}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Status")
        st.markdown(f'<div class="metric-card"><h4>Model</h4><p>✅ Ready</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()