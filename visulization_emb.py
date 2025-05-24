import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 设置为空字符串表示不可见任何 GPU
from pyhealth.medcode import InnerMap
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import SafeDrug
from model.safedrug_icd_unsup_pos import SafeDrug_ICD_Unsup_Pos
from pyhealth.tasks import drug_recommendation_mimic3_fn
from pyhealth.trainer import Trainer
from model.pcgrad_trainer import PCGrad_Trainer

from model.ood_splitter import data_shifting_split_by_visit, get_visit_info
from model.tasks import drug_recommendation_mimic3_all_visit_fn
from model.utils import get_all_tokens_from_samples
import pandas as pd
import torch
import random
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1994)

mimic3base = MIMIC3Dataset(
    #root=r"D:\Research\datasets\mimic-iii-clinical-database-1.4",
    root = 'data/mimic-iii-clinical-database-1.4',
    #root=r"F:\UGREEN\pc\Research\Datasets\mimic-iii-clinical-database-1.4",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    dev=False,
    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
)
mimic3base.stat()

#sample_dataset = mimic3base.set_task(drug_recommendation_mimic3_all_visit_fn) # Keep all vists rather than >=2 visits
sample_dataset = mimic3base.set_task(drug_recommendation_mimic3_fn)
sample_dataset.stat()


keys = ['conditions', 'procedures', 'drugs', 'drugs_hist']
condition_token = get_all_tokens_from_samples(sample_dataset.samples, keys[0])
procedure_token = get_all_tokens_from_samples(sample_dataset, keys[1])
drug_token = get_all_tokens_from_samples(sample_dataset, keys[2])
drug_hist_token = get_all_tokens_from_samples(sample_dataset, keys[3])


train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
print(f"train_dataset = {len(train_dataset)}, val_dataset = {len(val_dataset)}, test_dataset = {len(test_dataset)}")

train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

model = SafeDrug_ICD_Unsup_Pos(sample_dataset)
model.to("cpu")
#model = SafeDrug_ICDGraph_Prop(sample_dataset)
#model = SafeDrug_ICDGraph_Concate(sample_dataset)
#model = SafeDrug_ICD_Concate(sample_dataset)
#model = SafeDrug_ICD_Token(sample_dataset)
#model = SafeDrug_ICD(sample_dataset,)
#model = SafeDrug(sample_dataset,)

#model_name = model._get_name()
model_name = "SafeDrug_ICD_Unsup_Pos"
fig_path = "./figs/" + "SafeDrug_ICD_pos"

#ckpt_path = "output/20240513-safedrug-auc-0.717/best.ckpt"
#ckpt_path = "output/20240514-safedrug-unsup-pos-auc-0.7405/best.ckpt"
ckpt_path = "output/20240520-safedrug-pos-auc0.7243/best.ckpt"
state_dict = torch.load(ckpt_path)
model.load_state_dict(state_dict)

#fig_path = "./figs/" + model_name




if model_name == "SafeDrug":
    # tsne可视化embedding
    condition_embedding_table = model.embeddings["conditions"].weight.detach().cpu().numpy()[2:]
    procedure_embedding_table = model.embeddings["procedures"].weight.detach().cpu().numpy()[2:]

if model_name == "SafeDrug_ICD_Unsup_Pos":
    # tsne可视化embedding
    # condition_embedding_table = model.cond_graph_embedding.embedding.weight.detach().cpu().numpy()[2:]
    # procedure_embedding_table = model.proc_graph_embedding.embedding.weight.detach().cpu().numpy()[2:]

    
    condition_random_embedding = model.embeddings["conditions"].weight[2:]
    procedure_random_embedding = model.embeddings["procedures"].weight[2:]

    condition_pos_embedding = model.cond_pos_embedding(torch.arange(2, condition_random_embedding.shape[0]+2).long())
    procedure_pos_embedding = model.proc_pos_embedding(torch.arange(2, procedure_random_embedding.shape[0]+2).long())

    #condition_embedding_table = (condition_random_embedding + condition_pos_embedding).detach().cpu().numpy()
    #procedure_embedding_table = (procedure_random_embedding + procedure_pos_embedding).detach().cpu().numpy()

    condition_embedding_table = condition_pos_embedding.detach().cpu().numpy()
    procedure_embedding_table = procedure_pos_embedding.detach().cpu().numpy()

    

keys = ["<pad>", "<unk>"]
condition_token = model.feat_tokenizers["conditions"].vocabulary.idx2token
procedure_token = model.feat_tokenizers["procedures"].vocabulary.idx2token

# 提取第一位数据作为category
condition_label = [condition_token[i][0] for i in range(2,len(condition_token))]
procedure_label = [procedure_token[i][0] for i in range(2,len(procedure_token))]


le = LabelEncoder()
condition_label = le.fit_transform(condition_label)
procedure_label = le.fit_transform(procedure_label)


print(f"condition_embedding_table.shape {condition_embedding_table.shape},\
        procedure_embedding_table.shape {procedure_embedding_table.shape}")

tsne = TSNE(n_components=2)
condition_tsne_result = tsne.fit_transform(condition_embedding_table)
procedure_tsne_result = tsne.fit_transform(procedure_embedding_table)
palette = {}
for n, y in enumerate(set(condition_label)):
    palette[y] = f'C{n}'
palette2 = {}
for n, y in enumerate(set(procedure_label)):
    palette2[y] = f'C{n}'

plt.figure(figsize=(10, 10))
sns.scatterplot(x=condition_tsne_result.T[0], y=condition_tsne_result.T[1], hue=condition_label,palette= palette)
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.savefig(fig_path + "_condition_tsne.png", dpi=300)
print("saving to ", fig_path + "_condition_tsne.png")

plt.figure(figsize=(10, 10))
sns.scatterplot(x=procedure_tsne_result.T[0], y=procedure_tsne_result.T[1], hue=procedure_label,palette= palette2)
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.savefig(fig_path + "_procedure_tsne.png", dpi=300)
print("saving to ", fig_path + "_procedure_tsne.png")

print("finish")

    # # 根据不同condition_label 设置不同颜色
    # plt.scatter(condition_tsne_result[:, 0], condition_tsne_result[:, 1],marker='.',s=20, c=condition_label,cmap='tab20')
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    # #plt.show()
    # plt.savefig("./figs/safedrug_condition_tsne.png", dpi=300)

    # plt.scatter(procedure_tsne_result[:, 0], procedure_tsne_result[:, 1],marker='.',s=20)
    # #plt.show()
    # plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    # plt.savefig("./figs/safedrug_procedure_tsne.png", dpi=300)




exit(0)
trainer = Trainer(
#trainer = PCGrad_Trainer(
    model=model,
    #checkpoint_path="output/20240430-152513/best.ckpt",
    metrics=["jaccard_samples", "f1_samples", "pr_auc_samples", "ddi"],
)


trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    #test_dataloader=test_dataloader,
    epochs=25,
    monitor="pr_auc_samples",
    optimizer_params = {"lr": 1e-3},
)

print (trainer.evaluate(test_dataloader))








