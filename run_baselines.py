from pyhealth.medcode import InnerMap
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import SafeDrug, GAMENet, MICRON, MoleRec, RETAIN
from model.gamenet_icd import GAMENet_ICD
from model.micron_icd import MICRON_ICD
from model.molerec_icd import MoleRec_ICD
from model.retain_icd import RETAIN_ICD
from model.safedrug_icd_unsup_pos import SafeDrug_ICD_Unsup_Pos

from pyhealth.tasks import drug_recommendation_mimic3_fn , drug_recommendation_mimic4_fn

#from pyhealth.trainer import Trainer
from model.trainer import Trainer

from model.ood_splitter import data_shifting_split_by_visit, get_visit_info
from model.tasks import drug_recommendation_mimic3_all_visit_fn
from model.utils import get_all_tokens_from_samples

import torch
import random
import numpy as np
from datetime import datetime

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1994)


def prepare_drug_task_data(dataset="mimic3", dev=False):

    assert dataset in ["mimic3", "mimic4"]

    if dataset == "mimic3":
        mimic3base = MIMIC3Dataset(
            root = 'data/mimic-iii-clinical-database-1.4',
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=False,
            refresh_cache=False,
        )
        print("stat")
        mimic3base.stat()
        print("info")
        mimic3base.info()
        #sample_dataset = mimic3base.set_task(drug_recommendation_mimic3_all_visit_fn) # Keep all vists rather than >=2 visits
        mimic3_sample = mimic3base.set_task(drug_recommendation_mimic3_fn)

        return mimic3_sample

    if dataset == "mimic4":
        mimic4base = MIMIC4Dataset(
            root = 'data/mimic-iv-clinical-database-2.2/hosp',
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=dev,
        refresh_cache=False,
        )

        print("stat")
        mimic4base.stat()
        print("info")
        mimic4base.info()

        mimic4_sample = mimic4base.set_task(drug_recommendation_mimic4_fn)
        mimic4_sample.stat()
        print(mimic4_sample[0])

        return mimic4_sample


def get_dataloaders(mimic4_sample):
    train_dataset, val_dataset, test_dataset = split_by_patient(
        mimic4_sample, [0.8, 0.1, 0.1]
    )
    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    print(f"train_dataset = {len(train_dataset)}, val_dataset = {len(val_dataset)}, test_dataset = {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def train_evaluate(model, train_loader, val_loader, test_loader):
    trainer = Trainer(
        model=model,
        metrics=[
            "jaccard_samples",
            "f1_samples",
            "pr_auc_samples",
            "ddi",
            #"accuracy",
            # "hamming_loss",
            # "precision_samples",
            # "recall_samples",
        ],
        exp_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + model.__class__.__name__  + "_" + dataset 
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=25,
        monitor="pr_auc_samples",
        #optimizer_params = {"lr": 1e-3},
    )

    print (trainer.evaluate(test_loader))





if __name__ == "__main__":
    # experiment setting
    dataset = "mimic3"
    sample_dataset = prepare_drug_task_data(dataset=dataset)
    train_loader, val_loader, test_loader = get_dataloaders(sample_dataset)
    
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

    #model = SafeDrug(sample_dataset,)
    #model = SafeDrug_ICD_Unsup_Pos(sample_dataset)

    model = GAMENet_ICD(sample_dataset,) # done
    #model = GAMENet(sample_dataset,)

    #model = MICRON(sample_dataset)
    #model = MICRON_ICD(sample_dataset)

    #model = MoleRec(sample_dataset)
    # model = MoleRec_ICD(sample_dataset)

    # model = RETAIN(
    #     dataset=sample_dataset,
    #     feature_keys=[
    #         "conditions",
    #         "procedures",
    #     ],
    #     label_key="drugs",
    #     mode="multilabel",
    # )

    #model = RETAIN_ICD(
    #     dataset=sample_dataset,
    #     feature_keys=[
    #         "conditions",
    #         "procedures",
    #     ],
    #     label_key="drugs",
    #     mode="multilabel",
    # )

    train_evaluate(model, train_loader, val_loader,test_loader)