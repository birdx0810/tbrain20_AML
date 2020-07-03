import os


def filter_short_lines(lines, len_limit):
    """
    Read whole text, split tokens by spaces.
    Filter out tokens of lengths < `len_limit`.
    """
    tokens = []
    for line in lines:
        tokens += lines[0].strip('/n').split(' ')
    tokens = [token for token in tokens if len(token) > 25]
        
    return ' '.join(tokens)


def is_mojibake(text):
    """
    Detect if unusual word exists in `text`.

    Args:
    - `text` (str)
    """
    if text.find('\x96') > 0 or text.find('\x99') > 0:
        return True
    return False


def test():
    """
    list names of files containing mojibake
    """

    # config
    data_dir = '../data/cleaned_crawled_news'

    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    files = sorted(files)

    for file_name in files:
        with open(f'{data_dir}/{file_name}', 'r') as f:
            lines = f.readlines()

        lines = ' '.join(lines)
        if is_mojibake(lines):
            print(file_name)

    return

if __name__ == '__main__':
    test()
