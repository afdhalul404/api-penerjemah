from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import re

app = Flask(__name__)

# Load tokenizer from JSON for Tolaki to Indonesian
with open('tolaki-indonesia/tolaki_ina_tokenizer_input.json', 'r') as f:
    tokenizer_input_json = json.load(f)
    tokenizer_tolaki_to_indo = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_input_json)

with open('tolaki-indonesia/tolaki_ina_tokenizer_output.json', 'r') as f:
    tokenizer_output_json = json.load(f)
    tokenizer_indo = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_output_json)

# Load tokenizer from JSON for Indonesian to Tolaki
with open('indonesia-tolaki/ina_tolaki_tokenizer_input.json', 'r') as f:
    tokenizer_input_indo_json = json.load(f)
    tokenizer_indo_to_tolaki = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_input_indo_json)

with open('indonesia-tolaki/ina_tolaki_tokenizer_output.json', 'r') as f:
    tokenizer_output_tolaki_json = json.load(f)
    tokenizer_tolaki = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_output_tolaki_json)

vocab_size_tolaki_to_indo_input = len(tokenizer_tolaki_to_indo.word_index) + 1
vocab_size_tolaki_to_indo_output = len(tokenizer_indo.word_index) + 1
vocab_size_indo_to_tolaki_input = len(tokenizer_indo_to_tolaki.word_index) + 1
vocab_size_indo_to_tolaki_output = len(tokenizer_tolaki.word_index) + 1

inp_embed_size = 128
inp_lstm_cells = 256
tar_embed_size = 128
tar_lstm_cells = 256
attention_units = 256
batch_size = 1  # Adjust based on your model's expected batch size

length_input = 20  # Replace with the actual max length of input sequences
length_output = 20  # Replace with the actual max length of output sequences

# Define the Encoder class
class Encoder(tf.keras.Model):
    def __init__(self, inp_vocab_size, inp_embed_size, inp_lstm_cells, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_embedding = tf.keras.layers.Embedding(inp_vocab_size, inp_embed_size, trainable=True)
        self.lstm = tf.keras.layers.LSTM(inp_lstm_cells, return_sequences=True, return_state=True)

    def call(self, inp_sequence, hidden_sequence):
        emb_output = self.enc_embedding(inp_sequence)
        inp_lstm_output, state_h, state_c = self.lstm(emb_output, initial_state=hidden_sequence)
        return inp_lstm_output, state_h

    def initialize_hidden_states(self):
        return [tf.zeros([self.batch_size, inp_lstm_cells]), tf.zeros([self.batch_size, inp_lstm_cells])]

# Define the Attention class
class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(attention_units)
        self.W2 = tf.keras.layers.Dense(attention_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, hidden, output):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(hidden_with_time_axis) + self.W2(output)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Define the Decoder class
class Decoder(tf.keras.Model):
    def __init__(self, tar_embed_size, tar_vocab_size, tar_lstm_cells, attention_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_embedding = tf.keras.layers.Embedding(tar_vocab_size, tar_embed_size, trainable=True)
        self.lstm = tf.keras.layers.LSTM(tar_lstm_cells, return_sequences=True, return_state=True)
        self.attention = Attention(attention_units)
        self.final_layer = tf.keras.layers.Dense(tar_vocab_size)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        emb_output = self.dec_embedding(x)
        x_context = tf.concat([tf.expand_dims(context_vector, 1), emb_output], axis=-1)
        tar_lstm_output, tar_state_h, tar_state_c = self.lstm(x_context)
        tar_lstm_output_reshaped = tf.reshape(tar_lstm_output, (-1, tar_lstm_output.shape[2]))
        word_prob = self.final_layer(tar_lstm_output_reshaped)
        return word_prob, tar_state_h, attention_weights

# Instantiate and load the models for Tolaki to Indonesian
enc_model_tolaki_to_indo = Encoder(inp_vocab_size=vocab_size_tolaki_to_indo_input, inp_embed_size=inp_embed_size, inp_lstm_cells=inp_lstm_cells, batch_size=batch_size)
dec_model_tolaki_to_indo = Decoder(tar_embed_size=tar_embed_size, tar_vocab_size=vocab_size_tolaki_to_indo_output, tar_lstm_cells=tar_lstm_cells, attention_units=attention_units, batch_size=batch_size)

# Create a dummy input to build the model's structure before loading weights
dummy_input = tf.zeros((batch_size, length_input), dtype=tf.int32)
dummy_hidden = enc_model_tolaki_to_indo.initialize_hidden_states()
enc_model_tolaki_to_indo(dummy_input, dummy_hidden)
dec_model_tolaki_to_indo(tf.zeros((batch_size, 1), dtype=tf.int32), dummy_hidden[0], tf.zeros((batch_size, length_input, inp_lstm_cells)))

# Load weights for Tolaki to Indonesian
enc_model_tolaki_to_indo.load_weights('tolaki-indonesia/tolaki_ina_encoder.weights.h5')
dec_model_tolaki_to_indo.load_weights('tolaki-indonesia/tolaki_ina_decoder.weights.h5')

# Instantiate and load the models for Indonesian to Tolaki
enc_model_indo_to_tolaki = Encoder(inp_vocab_size=vocab_size_indo_to_tolaki_input, inp_embed_size=inp_embed_size, inp_lstm_cells=inp_lstm_cells, batch_size=batch_size)
dec_model_indo_to_tolaki = Decoder(tar_embed_size=tar_embed_size, tar_vocab_size=vocab_size_indo_to_tolaki_output, tar_lstm_cells=tar_lstm_cells, attention_units=attention_units, batch_size=batch_size)

# Create a dummy input to build the model's structure before loading weights
enc_model_indo_to_tolaki(dummy_input, dummy_hidden)
dec_model_indo_to_tolaki(tf.zeros((batch_size, 1), dtype=tf.int32), dummy_hidden[0], tf.zeros((batch_size, length_input, inp_lstm_cells)))

# Load weights for Indonesian to Tolaki
enc_model_indo_to_tolaki.load_weights('indonesia-tolaki/ina_tolaki_encoder.weights.h5')
dec_model_indo_to_tolaki.load_weights('indonesia-tolaki/ina_tolaki_decoder.weights.h5')

def preprocess_sentence(sentence, tokenizer, max_length):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    sentence = '<start> ' + sentence + ' <end>'
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded

def predict(input_sentence, tokenizer_input, tokenizer_output, enc_model, dec_model):
    inp_sequence = preprocess_sentence(input_sentence, tokenizer_input, length_input)
    enc_hidden = enc_model.initialize_hidden_states()
    enc_output, enc_hidden = enc_model(inp_sequence, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tokenizer_output.word_index.get('<start>', 1)], 0)  # Default to 1 if not found
    result = ''
    for t in range(length_output):
        predictions, dec_hidden, attention_weights = dec_model(dec_input, dec_hidden, enc_output)
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = tokenizer_output.index_word.get(predicted_id, '<unk>')  # Default to <unk> if not found
        
        # Check if predicted_word is '<end>' or 'end' and break if it is
        if predicted_word == '<end>' or predicted_word == 'end':
            break
        
        result += predicted_word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)
    
    # Remove any trailing 'end' if present
    result = re.sub(r'\bend\b', '', result).strip()
    
    return result

@app.route("/translate/tolaki-to-indo", methods=["POST"])
def translate_tolaki_to_indo():
    input_text = request.json.get("input_text")
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400
    try:
        result = predict(input_text, tokenizer_tolaki_to_indo, tokenizer_indo, enc_model_tolaki_to_indo, dec_model_tolaki_to_indo)
        return jsonify({"translation": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/translate/indo-to-tolaki", methods=["POST"])
def translate_indo_to_tolaki():
    input_text = request.json.get("input_text")
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400
    try:
        result = predict(input_text, tokenizer_indo_to_tolaki, tokenizer_tolaki, enc_model_indo_to_tolaki, dec_model_indo_to_tolaki)
        return jsonify({"translation": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
