import tiktoken

def compare_tokenization(text1: str, text2: str):
    # Get the tokenizer used by GPT-4
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Encode both texts
    tokens1 = encoding.encode(text1)
    tokens2 = encoding.encode(text2)
    
    print(f"--- Text 1: '{text1}' ---")
    print(f"Number of tokens: {len(tokens1)}")
    # Print each token and its decoded string
    for t in tokens1:
        print(f"Token ID: {t} -> String: '{encoding.decode([t])}'")
        
    print(f"\n--- Text 2: '{text2}' ---")
    print(f"Number of tokens: {len(tokens2)}")
    for t in tokens2:
        print(f"Token ID: {t} -> String: '{encoding.decode([t])}'")

if __name__ == "__main__":
    # A standard word vs a rare medical word
    standard_text = "apple"
    medical_text = "otorhinolaryngology"
    
    compare_tokenization(standard_text, medical_text)
