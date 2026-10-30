import re

def extract_label_from_response(response, label_choices):
    """Extract classification label from model response.
    
    Args:
        response: Raw model output text
        label_choices: List of possible label values
    
    Returns:
        Index of the matched label, or None if no match found
    """
    response = response.strip()
    if label_choices == ['Yes', 'No', 'Maybe']:
        response = response.lower()
        if response == 'yes':
            return 0
        elif response == 'no':
            return 1
        elif response == 'maybe':
            return 2
        return None
    response = response.lower()
    for (i, label) in enumerate(label_choices):
        if label.lower().strip() in response:
            return i
    return None

def extract_answer_from_response(response):
    """Extract answer option (A-E) from model response.
    
    Args:
        response: Raw model output text
    
    Returns:
        Integer index (0-4) corresponding to answer A-E, or None if no match found
    """
    response = response.strip().lower()
    if response.startswith('a'):
        return 0
    elif response.startswith('b'):
        return 1
    elif response.startswith('c'):
        return 2
    elif response.startswith('d'):
        return 3
    elif response.startswith('e'):
        return 4
    if response.startswith('1') or response == '0':
        return 0
    elif response.startswith('2'):
        return 1
    elif response.startswith('3'):
        return 2
    elif response.startswith('4'):
        return 3
    elif response.startswith('5'):
        return 4
    for (letter, idx) in [('a', 0), ('b', 1), ('c', 2), ('d', 3), ('e', 4)]:
        if letter in response:
            return idx
    for (num, idx) in [('1', 0), ('2', 1), ('3', 2), ('4', 3), ('5', 4)]:
        if num in response:
            return idx
    return None
