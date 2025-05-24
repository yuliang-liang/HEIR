from typing import Dict, List

def flatten_list(l: List) -> List:
    """Flattens a list of list.

    Args:
        l: List, the list of list to be flattened.

    Returns:
        List, the flattened list.

    Examples:
        >>> flatten_list([[1], [2, 3], [4]])
        [1, 2, 3, 4]R
        >>> flatten_list([[1], [[2], 3], [4]])
        [1, [2], 3, 4]
    """
    assert isinstance(l, list), "l must be a list."
    return sum(l, [])


def get_all_tokens_from_samples(samples,
    key: str, remove_duplicates: bool = True, sort: bool = True) -> List[str]:
    """Gets all tokens with a specific key in the samples.

    Args:
        samples: list or sample_dataset etc.
        key: the key of the tokens in the samples.
        remove_duplicates: whether to remove duplicates. Default is True.
        sort: whether to sort the tokens by alphabet order. Default is True.

    Returns:
        tokens: a list of tokens.
    """
    input_info= {'visit_id': {'type':'str', 'dim': 0},
                'patient_id': {'type': 'str', 'dim': 0},
                'conditions': {'type': 'str', 'dim': 3},
                'procedures': {'type': 'str', 'dim': 3}, 
                'drugs': {'type': 'str', 'dim': 2},
                'drugs_hist': {'type': 'str', 'dim': 3}
                }
    assert key in input_info.keys(), f"key {key} not in input_info"

    input_type = input_info[key]["type"]
    input_dim = input_info[key]["dim"]

    tokens = []
    for sample in samples:
        if input_dim == 0:
            # a single value
            tokens.append(sample[key])
        elif input_dim == 2:
            # a list of codes
            tokens.extend(sample[key])
        elif input_dim == 3:
            # a list of list of codes
            tokens.extend(flatten_list(sample[key]))
        else:
            raise NotImplementedError
    if remove_duplicates:
        tokens = list(set(tokens))
    if sort:
        tokens.sort()
    return tokens