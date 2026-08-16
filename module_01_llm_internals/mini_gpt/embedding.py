import numpy as np

def positional_encoding_demo():
    # Let's use small values so we can print and visualize every single number!
    vocab_size = 10
    d_model = 4        # Embedding dimension (each token is represented by 4 numbers)
    seq_len = 3        # Sequence length (we have 3 words: "the", "cat", "sat")
    
    # 1. TOKEN EMBEDDING TABLE (vocab_size x d_model)
    # In a real model, these weights are learned during training.
    # We'll initialize them with random numbers for this demo.
    np.random.seed(42)
    embedding_table = np.random.uniform(-1, 1, size=(vocab_size, d_model))
    
    print("=== 1. EMBEDDING TABLE (vocab_size x d_model) ===")
    print(f"Shape: {embedding_table.shape}")
    for i in range(vocab_size):
        print(f"Token ID {i} vector: {embedding_table[i].round(2)}")
    print("=" * 50 + "\n")
    
    # Let's say our input sentence is: "the cat sat"
    # Tokenized input IDs:
    input_ids = [2, 5, 7]  # "the" = 2, "cat" = 5, "sat" = 7
    print(f"Input sentence token IDs: {input_ids}\n")
    
    # 2. EMBEDDING LOOKUP
    # We grab the vectors corresponding to our input IDs from the table.
    token_embeddings = embedding_table[input_ids]
    print("=== 2. TOKEN EMBEDDINGS (seq_len x d_model) ===")
    print(f"Shape: {token_embeddings.shape}")
    print(f"Pos 0 (Token 2): {token_embeddings[0].round(2)}")
    print(f"Pos 1 (Token 5): {token_embeddings[1].round(2)}")
    print(f"Pos 2 (Token 7): {token_embeddings[2].round(2)}")
    print("=" * 50 + "\n")
    
    # 3. POSITIONAL ENCODINGS (seq_len x d_model)
    # We create a unique vector for each position index (0, 1, 2).
    # We will use a simple fixed sinusoidal wave encoding.
    pos_encodings = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                pos_encodings[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                pos_encodings[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
                
    print("=== 3. POSITIONAL ENCODINGS (seq_len x d_model) ===")
    print(f"Shape: {pos_encodings.shape}")
    print(f"Pos 0 Encoder: {pos_encodings[0].round(2)}")
    print(f"Pos 1 Encoder: {pos_encodings[1].round(2)}")
    print(f"Pos 2 Encoder: {pos_encodings[2].round(2)}")
    print("=" * 50 + "\n")
    
    # 4. SUM THE TWO MATRICES
    # This is where they work together. We add them element-by-element!
    final_embeddings = token_embeddings + pos_encodings
    
    print("=== 4. FINAL EMBEDDINGS (Token + Position) ===")
    print(f"Shape: {final_embeddings.shape}")
    print(f"Final Pos 0 vector: {final_embeddings[0].round(2)}  <- (Token vector {token_embeddings[0].round(2)} + Pos vector {pos_encodings[0].round(2)})")
    print(f"Final Pos 1 vector: {final_embeddings[1].round(2)}  <- (Token vector {token_embeddings[1].round(2)} + Pos vector {pos_encodings[1].round(2)})")
    print(f"Final Pos 2 vector: {final_embeddings[2].round(2)}  <- (Token vector {token_embeddings[2].round(2)} + Pos vector {pos_encodings[2].round(2)})")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    positional_encoding_demo()
