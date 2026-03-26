import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def build_vocab(texts, max_words=5000):
    freq = {}
    for text in texts:
        for word in text.split():
            freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)[:max_words]
    return {w:i+1 for i,w in enumerate(sorted_words)}

def text_to_sequence(text, word_index, max_len=50, vocab_size=5000):
    seq = [word_index.get(w, 0) for w in text.split()]
    seq = seq[:max_len] + [0]*(max_len-len(seq))
    return seq