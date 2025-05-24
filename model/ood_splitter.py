from itertools import chain
from typing import Optional, Tuple, Union, List
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pyhealth.datasets import SampleBaseDataset, BaseEHRDataset

# TODO: train_dataset.dataset still access the whole dataset which may leak information
# TODO: add more splitting methods


from pyhealth.datasets import BaseEHRDataset


def get_visit_info(base_data: BaseEHRDataset):
    # get visit level info from mimic3base
    info_list = []
    for patient_id, patient in tqdm(base_data.patients.items(), desc='get visit level info'):
        for visit_id, visit in patient.visits.items():
            # ecnounter_time
            # hour = visit.encounter_time.hour
            # month = visit.encounter_time.month
            # week = visit.encounter_time.weekday()

            d = {
                    'patient_id': patient_id,
                    'visit_id': visit_id,
                    # 'hour': hour,
                    # 'week': week,
                    # 'month': month,
                    'encounter_time':visit.encounter_time,
                    'gender':patient.gender,
                    'birth':patient.birth_datetime,
                    'ethnicity_from_patient':patient.ethnicity,
                    }
            # attr_dict {'insurance': 'Medicare', 'language': nan, 'religion': 'JEWISH', 
            #'marital_status': 'MARRIED', 'ethnicity': 'WHITE'}
            d.update(visit.attr_dict)
            
            info_list.append(d)
            
    visit_info_pd = pd.DataFrame(info_list)
    return visit_info_pd



#Date shifting including time, week and month
def data_shifting_split_by_visit(
    dataset: SampleBaseDataset,
    mimic3base:BaseEHRDataset,
    ratios: Optional[Union[Tuple[float, float, float], List[float]]] = [0.8, 0.1, 0.1],
    shift: Optional[str] = 'time_of_day',
    method: Optional[str] = 'leave_one_domain_out',
    seed: Optional[int] = None,
):
    """Splits the dataset by visit (i.e., samples).

    Args:
        dataset: a `SampleBaseDataset` object
        ratios: a list/tuple of ratios for train / val / test
        shift: the shifting method, must be one of 'time', 'week', 'month'
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    #assert shift in ['time', 'week', 'month'], "method must be one of 'time', 'week', 'month'"
    
    # TODO: implement data shifting based on the selected method
        
    visit_info_pd = get_visit_info(mimic3base)

    if shift == 'time_of_day':
        # In MIMIC III, Time of day - a measurement made at 15:00:00 was actually made at 15:00:00 local standard time.
        df = visit_info_pd[['visit_id','encounter_time']]
        df['hour'] = df['encounter_time'].dt.hour
        #df['hour'] = df['hour'].apply(lambda x: x//2) #每两个小时为一个domain
        df_group = df.groupby(['hour'])

    elif shift == 'day_of_week':
        # Day of the week - a measurement made on a Sunday will appear on a Sunday in the future.
        df = visit_info_pd[['visit_id','encounter_time']]
        df['weekday'] = df['encounter_time'].dt.weekday
        df_group = df.groupby(['weekday'])


    elif shift == 'season':
        # Seasonality - a measurement made during the winter months will appear during a winter month.
        df = visit_info_pd[['visit_id','encounter_time']]
        df['month'] = df['encounter_time'].dt.month
        df_group = df.groupby(['month'])

    #domain_to_visit {domain:[visit_id, visit_id, ...]}
    domain_to_visit = df_group['visit_id'].apply(list).to_dict()
    num_domains = len(domain_to_visit)

    #TODO 这个是domain vocab 暂时没用
    domain_voc = list(domain_to_visit.keys())

    #根据dataset.visit_to_index  构造domain_to_index字典{domain:[index, index, ...]}
    domain_to_index = {domain: [] for domain in domain_voc}
    # 遍历 visit_to_index字典{v_id:index}, 为每个visit找到对应的domain
    for visit, index in tqdm(dataset.visit_to_index.items(), desc=f'split visit by domain ({shift})'):
        # look up the domain of the visit
        for domain, visits in domain_to_visit.items():
            if visit in visits:
                domain_to_index[domain].append(index[0])
                break

    if method == 'all':
        # return all information with domain info for development
        ret={
            'shift':shift,
            'visit_info_pd':visit_info_pd,
            'num_domains': num_domains,
            'domain_voc': domain_voc,
            'domain_to_visit': domain_to_visit,
            'domain_to_index': domain_to_index,
            'visit_to_index': dataset.visit_to_index
            }
        return ret
    # 返回每个domain的数据
    elif method == 'split_by_domain':
        # each domain is a dataset
        ood_dataset_dict = {}
        for i in range(num_domains):
            train_domain = [i]
            valtest_domain = [i]
            all_index = domain_to_index[i]
            np.random.shuffle(all_index)
            train_index = all_index[: int(len(all_index) * ratios[0])]
            val_index = all_index[int(len(all_index) * ratios[0]) : int(len(all_index) * (ratios[0] + ratios[1]))]
            test_index = all_index[int(len(all_index) * (ratios[0] + ratios[1])) :]
            train_dataset = torch.utils.data.Subset(dataset, train_index)
            val_dataset = torch.utils.data.Subset(dataset, val_index)
            test_dataset = torch.utils.data.Subset(dataset, test_index)
            ood_dataset_dict[i] = {
                'train_domain': train_domain,
                'valtest_domain': valtest_domain,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset
            }
        return ood_dataset_dict
        
    elif method == 'leave_one_domain_out':
        # leave_one_domain_out method
        ood_dataset_dict = {}
        for i in range(num_domains):
            train_domain= domain_voc[:i] + domain_voc[i+1:]
            valtest_domain = [domain_voc[i]]
            train_index = list(
                chain(*[domain_to_index[d] for d in train_domain])
                )
            valtest_index = list(
                chain(*[domain_to_index[d] for d in valtest_domain])
                )
            # the ratio of val and test is 1:1
            val_index = valtest_index[: int(len(valtest_index) *0.5)]
            test_index = valtest_index[int(len(valtest_index) *0.5):]

            train_dataset = torch.utils.data.Subset(dataset, train_index)
            val_dataset = torch.utils.data.Subset(dataset, val_index)
            test_dataset = torch.utils.data.Subset(dataset, test_index)

            ood_dataset_dict[i] = {
                'train_domain': train_domain,
                'valtest_domain': valtest_domain,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset
            }


        return ood_dataset_dict

    




def split_by_visit(
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by visit (i.e., samples).

    Args:
        dataset: a `SampleBaseDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    train_index = index[: int(len(dataset) * ratios[0])]
    val_index = index[
        int(len(dataset) * ratios[0]) : int(len(dataset) * (ratios[0] + ratios[1]))
    ]
    test_index = index[int(len(dataset) * (ratios[0] + ratios[1])) :]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def split_by_patient(
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
):
    """Splits the dataset by patient.

    Args:
        dataset: a `SampleBaseDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(dataset.patient_to_index.keys())
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)
    train_patient_indx = patient_indx[: int(num_patients * ratios[0])]
    val_patient_indx = patient_indx[
        int(num_patients * ratios[0]) : int(num_patients * (ratios[0] + ratios[1]))
    ]
    test_patient_indx = patient_indx[int(num_patients * (ratios[0] + ratios[1])) :]
    train_index = list(
        chain(*[dataset.patient_to_index[i] for i in train_patient_indx])
    )
    val_index = list(chain(*[dataset.patient_to_index[i] for i in val_patient_indx]))
    test_index = list(chain(*[dataset.patient_to_index[i] for i in test_patient_indx]))
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset


def split_by_sample(
    dataset: SampleBaseDataset,
    ratios: Union[Tuple[float, float, float], List[float]],
    seed: Optional[int] = None,
    get_index: Optional[bool] = False,
):
    """Splits the dataset by sample

    Args:
        dataset: a `SampleBaseDataset` object
        ratios: a list/tuple of ratios for train / val / test
        seed: random seed for shuffling the dataset

    Returns:
        train_dataset, val_dataset, test_dataset: three subsets of the dataset of
            type `torch.utils.data.Subset`.

    Note:
        The original dataset can be accessed by `train_dataset.dataset`,
            `val_dataset.dataset`, and `test_dataset.dataset`.
    """
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    train_index = index[: int(len(dataset) * ratios[0])]
    val_index = index[
                int(len(dataset) * ratios[0]): int(
                    len(dataset) * (ratios[0] + ratios[1]))
                ]
    test_index = index[int(len(dataset) * (ratios[0] + ratios[1])):]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    
    if get_index:
        return torch.tensor(train_index), torch.tensor(val_index), torch.tensor(test_index)
    else:
        return train_dataset, val_dataset, test_dataset
    


if __name__ == "__main__":

    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.datasets import split_by_patient, get_dataloader
    from pyhealth.models import SafeDrug
    from pyhealth.tasks import drug_recommendation_mimic3_fn
    from pyhealth.trainer import Trainer
    mimic3base = MIMIC3Dataset(
        root=r"D:\Research\datasets\mimic-iii-clinical-database-1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    )
    mimic3base.stat()

    # STEP 2: set task
    sample_dataset = mimic3base.set_task(drug_recommendation_mimic3_fn)
    sample_dataset.stat()

    ood_dataset_dict = data_shifting_split_by_visit(
        sample_dataset,mimic3base, shift='time_of_day', method='all'
    )
    
    # the number of train/val/test datasets
    # for domain, data in ood_dataset_dict.items():
    #     print(domain, len(data['train_dataset']), len(data['val_dataset']), len(data['test_dataset']))  
    #     print(f"train_domain: {data['train_domain']}")
    #     print(f"valtest_domain: {data['valtest_domain']}")
    #     print('--------------------------------------------')
    #     print(f'train={len(data["train_dataset"])} val={len(data["val_dataset"])} test={len(data["test_dataset"])}')