import sys

#sys.path.append("..")

from pyhealth.models import *
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets.utils import collate_fn_dict
from pyhealth.trainer import Trainer
# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.ensemble import RandomForestClassifier as RF
# from sklearn.neural_network import MLPClassifier as NN

import warnings

warnings.filterwarnings("ignore")

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import SafeDrug, GAMENet
from model.safedrug_icd import SafeDrug_ICD
from model.safedrug_icd_token import SafeDrug_ICD_Token
from pyhealth.tasks import drug_recommendation_mimic3_fn
from pyhealth.trainer import Trainer

from model.ood_splitter import data_shifting_split_by_visit, get_visit_info
from model.tasks import drug_recommendation_mimic3_all_visit_fn
from model.utils import get_all_tokens_from_samples
import pandas as pd
import torch
import random
import numpy as np
from tqdm import tqdm

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
dataset = mimic3base.set_task(drug_recommendation_mimic3_fn)
dataset.stat()



train_dataset, val_dataset, test_dataset = split_by_patient(dataset, [0.8, 0.1, 0.1])
print(f"train_dataset = {len(train_dataset)}, val_dataset = {len(val_dataset)}, test_dataset = {len(test_dataset)}")

train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False) 


#model = GAMENet(sample_dataset,)

# STEP 4: define trainer
# trainer = Trainer(
#     model=model,
#     #checkpoint_path="output/20240429-212453/best.ckpt",
#     metrics=["jaccard_samples", "f1_samples", "pr_auc_samples", "ddi"],
# )

# trainer.train(
#     train_dataloader=train_dataloader,
#     val_dataloader=val_dataloader,
#     epochs=25,
#     monitor="pr_auc_samples",
# )

# print (trainer.evaluate(test_dataloader))



def get_metrics_result(mode, y_gt, y_pred, y_prob):
    from pyhealth.metrics import multilabel_metrics_fn
    metrics_ =["jaccard_samples", "f1_samples", "pr_auc_samples", "ddi"]
    if mode == "multilabel":
        scores = multilabel_metrics_fn(y_gt, y_pred,metrics=metrics_)
    for key in scores.keys():
        print("{}: {:.4f}".format(key, scores[key]))

    return scores

if __name__ == "__main__":
    from model.LR import LR
    from model.ECC import ECC
    #classic_ml_models = [LR(), RF, NN, LR(solver='lbfgs')]
    tables_ = ["conditions", "procedures"]
    target_ = "drugs"
    mode_ = "multilabel"
    val_metric = "pr_auc"


    model = ECC(
        dataset=dataset,
        tables=tables_,
        target=target_,
        mode=mode_,
    )

    model.fit(
        train_loader=train_dataloader,
        reduce_dim=100,
        val_loader=val_dataloader,
        val_metric=val_metric,
    )

    y_true_all = []
    y_prob_all = []
    y_pred_all = []

    for data in tqdm(test_dataloader, desc="Evaluation"):
            output = model(**data)
            y_true = output["y_true"]
            y_prob = output["y_prob"]
            y_pred = output["y_pred"]
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            y_pred_all.append(y_pred)
    y_gt_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)    
    #y_gt, y_prob, y_pred = evaluate(model, test_dataloader)

    scores = get_metrics_result(mode_, y_gt_all, y_pred_all, y_prob_all)
    
    print("see you again!")