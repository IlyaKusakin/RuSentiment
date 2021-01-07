import razdel
import transformers
import torch

def filter_text(text):
    line = razdel.tokenize(text.lower())
    line = [token.text for token in line]
    filtered_line = ' '.join([token for token in line if token.isalpha()])
        
    return filtered_line

def sample_batch(text, max_length=16):
    tokenizer = transformers.BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    tokenizer_output = tokenizer.encode_plus(
            filter_text(text), max_length = max_length,
            return_tensors = 'pt', padding = 'max_length')
        
    return {
            "input_ids": tokenizer_output['input_ids'].squeeze(0)[:max_length], 
            "mask": tokenizer_output['attention_mask'].squeeze(0)[:max_length],
           }

def predict(texts, model, device):
    inputs = []
    mask = []
    for text in texts:
        batch = sample_batch(text)
        inputs.append(batch['input_ids'].numpy().tolist())
        mask.append(batch['mask'].numpy().tolist())

    inputs = torch.tensor(inputs)
    mask = torch.tensor(mask)
    model.eval()
    with torch.no_grad():
      outputs = model(input_ids = inputs.to(device), attention_mask = mask.to(device))
      probas = torch.nn.functional.softmax(outputs.logits, dim=-1)

    result = []
    for proba in probas:
        result.append({"negative": proba.tolist()[0], "positive": proba.tolist()[1]})
    
    return result

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_tokenizer = transformers.BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
tuned_RuBERT = transformers.BertForSequenceClassification.from_pretrained(
        'DeepPavlov/rubert-base-cased', 
        output_attentions=True,
        pad_token_id=bert_tokenizer.eos_token_id,
        num_labels=2,
        return_dict = True
        ).to(device)

weights_file = 'weights/tuned_RuBERT_common.pt'
tuned_RuBERT.load_state_dict(torch.load(weights_file))

messages = ['черт я так устала не могу', 'ура я довольный и счастливый']
results = predict(messages, tuned_RuBERT, device)

for message, sentiment in zip(messages, results):
    # черт я так устала не могу -> {'negative': 0.985421359539032, 'positive': 0.014578700996935368}
    # ура я довольный и счастливый -> {'negative': 0.015193315222859383, 'positive': 0.9848067164421082}
    print(message, '->', sentiment)

