import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    with open("shakespeare.txt") as f: 
        text = f.read()

    tokens = enc.encode_ordinary(text)
    tokens.append(enc.eot_token)

    train_tokens = tokens[:int(0.95*len(tokens))] # 95% of the data for training
    val_tokens = tokens[int(0.95*len(tokens)):] # 5% of the data for validation
    
    np.save("shakespeare_train.npy", np.array(train_tokens, dtype=np.uint16))
    np.save("shakespeare_val.npy", np.array(val_tokens, dtype=np.uint16))
