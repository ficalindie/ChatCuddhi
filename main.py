from transformers import (
    BloomPreTrainedModel, BloomModel, BloomTokenizerFast, 
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoTokenizer
    )
from dataload import DataCreator
from ChatData import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch

def train(chatData, model, optim):
    epochs = 20

    for _ in tqdm.tqdm(range(epochs)):
        for X, a in chatData:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), "model_state.pt")
        print(infer("hello how are you"))


def infer(inp):
    inp = "<startofstring> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")

    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)

    output = model.generate(X, attention_mask=a)
    output = tokenizer.decode(output[0])
    return output

paths = [
    ["C:/Users/marci/Desktop/ficalindie/ChatCuddhi/data/words_files/gr",
    "C:/Users/marci/Desktop/ficalindie/ChatCuddhi/data/words_files/it"],
    ["C:/Users/marci/Desktop/ficalindie/ChatCuddhi/data/corrected/tgt_griko",
    "C:/Users/marci/Desktop/ficalindie/ChatCuddhi/data/corrected/src2_italian"]
] 

prompt_formula = [
    "per favore traduci in griko: ", 
    "traduci: ", 
    "come si dice in griko: ", 
    "traduzione in griko di: ", 
    ""
]

output_dir = "chat_data.json"

datagenerator = DataCreator(
    prompt_templates=prompt_formula, 
    dirs=paths, 
    out_dir=output_dir
    )

datagenerator.save_data()

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                "bos_token": "<startofstring>",
                                "eos_token": "<endofstring>"})
tokenizer.add_tokens(["<bot>:"])

model = BloomModel.from_pretrained("bigscience/bloom-560m")
#model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

# print(tokenizer.decode(model.generate(**tokenizer("hey i was good at basketball but ",
#                          return_tensors="pt"))[0]))

chatData = ChatData("chat_data.json", tokenizer)
chatData =  DataLoader(chatData, batch_size=64)

model.train()

optim = Adam(model.parameters(), lr=1e-3)

print("training .... ")
train(chatData, model, optim)

print("infer from model : ")
while True:
  inp = input()
  print(infer(inp))