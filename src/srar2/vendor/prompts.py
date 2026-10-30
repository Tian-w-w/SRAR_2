from __future__ import annotations

def overlap(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    indices = [i for (i, (a, b)) in enumerate(offsets) if b > start and a < end]
    if not indices:
        raise RuntimeError(f'offset mapping failure: {start}:{end}')
    return (indices[0], indices[-1] + 1)

def anchors(start: int, end: int) -> tuple[dict[str, int], bool]:
    width = end - start
    if width < 1:
        raise RuntimeError('empty candidate span')
    points = [start + min(width - 1, max(0, width * k // 6)) for k in (1, 3, 5)]
    return ({'RE': points[0], 'RM': points[1], 'RL': points[2]}, len(set(points)) < 3)

def prompt_and_chars(task: str, row: dict) -> tuple[str, tuple[int, int], tuple[int, int], str]:
    if task == 'sst2':
        prefix = 'Classify the sentence into one of the following categories: positive or negative.\nSentence: '
        text = prefix + row['sentence'] + '\nAnswer:'
        return (text, (len(prefix), len(prefix) + len(row['sentence'])), (0, len(prefix)), 'official classification instruction prefix')
    if task == 'boolq':
        prefix = 'Read the text and answer the question by True or False.\nText: '
        text = prefix + row['passage'] + '\nQuestion: ' + row['question'] + '?\nAnswer:'
        c1 = len(prefix) + len(row['passage'])
        q0 = c1 + len('\nQuestion: ')
        return (text, (len(prefix), c1), (q0, q0 + len(row['question'])), 'official question field')
    text = row['context'] + '\n' + row['question'] + '\n' + '\n'.join((f'{chr(65 + i)}. {choice}' for (i, choice) in enumerate(row['options']))) + '\nAnswer:'
    q0 = len(row['context']) + 1
    return (text, (0, len(row['context'])), (q0, q0 + len(row['question'])), 'official LogiQA query field; options excluded')
