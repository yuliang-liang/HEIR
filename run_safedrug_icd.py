import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 设置为空字符串表示不可见任何 GPU
from pyhealth.medcode import InnerMap
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import SafeDrug

from model.safedrug_icd_unsup_pos import SafeDrug_ICD_Unsup_Pos
#from model.safedrug_icd_unsup_pos_0527 import SafeDrug_ICD_Unsup_Pos
from pyhealth.tasks import drug_recommendation_mimic3_fn
#from pyhealth.trainer import Trainer
from model.trainer import Trainer
#from model.pcgrad_trainer import PCGrad_Trainer

from model.ood_splitter import data_shifting_split_by_visit, get_visit_info
from model.tasks import drug_recommendation_mimic3_all_visit_fn
from model.utils import get_all_tokens_from_samples
import pandas as pd
import torch
import random
import numpy as np

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1994)


mimic3base = MIMIC3Dataset(
    root = 'data/mimic-iii-clinical-database-1.4',
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    dev=False,
    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
)
mimic3base.stat()

#sample_dataset = mimic3base.set_task(drug_recommendation_mimic3_all_visit_fn) # Keep all vists rather than >=2 visits
sample_dataset = mimic3base.set_task(drug_recommendation_mimic3_fn)
sample_dataset.stat()

# statistics 
from pyhealth.datasets.utils import flatten_list
max_visit = max([len(sample) for sample in sample_dataset.patient_to_index.values()])
num_events = [len(flatten_list(sample['conditions'])) for sample in sample_dataset.samples]
max_condition = max(num_events)
num_events = [len(flatten_list(sample['procedures'])) for sample in sample_dataset.samples]
max_procedure = max(num_events)
num_events = [len(sample['drugs']) for sample in sample_dataset.samples]
max_drug = max(num_events)

print(f"max_visit = {max_visit}, max_condition = {max_condition}, max_procedure = {max_procedure}, max_drug = {max_drug}")



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



trainer = Trainer(
    model=model,
    #checkpoint_path="output/20240430-152513/best.ckpt",
    metrics=["jaccard_samples", "f1_samples", "pr_auc_samples", "ddi"],
)



trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    epochs=25,
    monitor="pr_auc_samples",
    optimizer_params = {"lr": 1e-3},
)

print (trainer.evaluate(test_dataloader))

exit(0)

if model._get_name() == "SafeDrug":
    # tsne可视化embedding
    condition_embedding_table = model.embeddings["conditions"].weight.detach().cpu().numpy()
    procedure_embedding_table = model.embeddings["procedures"].weight.detach().cpu().numpy()
    condition_token = model.feat_tokenizers["conditions"].vocabulary.idx2token
    procedure_token = model.feat_tokenizers["procedures"].vocabulary.idx2token
    condition_label = [condition_token[i][0] for i in range(len(condition_token))]
    # 将condition_label的字符转换为数字
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    condition_label = le.fit_transform(condition_label)

    print(condition_embedding_table.shape, procedure_embedding_table.shape)
    #print(condition_token)
    #print(procedure_token)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2)
    condition_tsne_result = tsne.fit_transform(condition_embedding_table)
    procedure_tsne_result = tsne.fit_transform(procedure_embedding_table)
    # 根据不同condition_label 设置不同颜色
    plt.scatter(condition_tsne_result[:, 0], condition_tsne_result[:, 1],marker='.',s=20, c=condition_label,cmap='tab20')
    plt.show()
    plt.savefig("./figs/condition_tsne.png", dpi=300)
    pass

if model._get_name() == "SafeDrug_ICD":
    condition_embedding_table = model.cond_dense.weight.detach().cpu().numpy().T
    #print(condition_embedding_table.shape, procedure_embedding_table.shape)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2)
    condition_tsne_result = tsne.fit_transform(condition_embedding_table)
    #procedure_tsne_result = tsne.fit_transform(procedure_embedding_table)
    plt.scatter(condition_tsne_result[:, 0], condition_tsne_result[:, 1],marker='.',s=20)
    plt.show()
    plt.savefig("./figs/safedrug_icd_condition_tsne", dpi=300)
    pass

if model._get_name == "SafeDrug_ICD":
    # tsne可视化embedding
    condition_embedding_table = model.cond_embedding.weight.detach().cpu().numpy()
    procedure_embedding_table = model.proc_embedding.weight.detach().cpu().numpy()

    print(condition_embedding_table.shape, procedure_embedding_table.shape)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2)
    condition_tsne_result = tsne.fit_transform(condition_embedding_table)
    procedure_tsne_result = tsne.fit_transform(procedure_embedding_table)
    plt.scatter(condition_tsne_result[:, 0], condition_tsne_result[:, 1],marker='.',s=20)
    plt.show()
    plt.savefig("./figs/safedrug_icd_token_condition_tsne", dpi=300)







