from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("pretrain_model/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",local_files_only=True)
model = BertModel.from_pretrained("pretrain_model/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",local_files_only=True)

inputs = tokenizer("hello world", return_tensors='pt', max_length=512, truncation=True, padding=True)['input_ids']
# [1, squence, hidden_dim]
last_hidden_state = model(inputs).last_hidden_state
cls_token = last_hidden_state[:, 0, :]

exit()


tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",cache_dir='./')
model = AutoModelForMaskedLM.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",cache_dir='./')
text = 'The <mask> is the main cause of death in the world'
inputs = tokenizer(text, return_tensors='pt')
last_hidden_state = model(**inputs).last_hidden_state
print(last_hidden_state)

pass
