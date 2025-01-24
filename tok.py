from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
with open("data/Ant-Man.txt", "r") as f:
    text = f.read()
    print(len(text))
    print(text[4])
    t = tokenizer.encode(text, return_tensors="pt")
    f.close()

print(t.shape)
print(t[0][4])
