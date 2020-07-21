def decode(predictions, test_data, raw_labels, tokenizer):
    r = []
    l = []
    p = []

    for index, row in enumerate(predictions):
        t_row = test_data[index][0]

        tmp_i = 0
        t_label = []
        tmp = []
        for i, char in enumerate(test_data[index][0]):

            if test_data[index][1][i] == 1 and tmp_i == i-1:
                # print("Up")
                tmp.append(char.item())
                tmp_i = i

            elif test_data[index][1][i] == 1 and tmp_i == 0:
                # print("Mid")
                tmp.append(char.item())
                tmp_i = i

            elif test_data[index][1][i] == 1 and tmp_i != i-1:
                # print("Down")
                t_label.append(tmp)
                tmp = []
                tmp.append(char.item())
                tmp_i = i

        if tmp != []:
            t_label.append(tmp)

        pred = []
        tmp = []
        tmp_idx = None
        for idx, prob in row:
            if tmp_idx != idx-1 and tmp_idx is not None:
                pred.append(tmp)
                tmp = []

            tmp.append(t_row[idx].item())
            tmp_idx = idx

        pred.append(tmp)

        # print(f"{'-'*50}")
        # print(f"{index}|Raw:    \t{raw_labels[index]}")
        # print(f"{index}|Label:  \t{list(set([''.join(n) for n in t.decode(t_label) if n != []]))}")
        # print(f"{index}|Decoded:\t{list(set([''.join(n) for n in t.decode(row) if n != []]))}")
        r.append(raw_labels[index])
        l.append(list(set([''.join(n) for n in tokenizer.decode(t_label) if n != []])))
        p.append(list(set([''.join(n) for n in tokenizer.decode(row) if n != []])))

    return r, l, p

if __name__ == "__main__":
    import opencc
    converter = opencc.OpenCC('s2t.json')
    with open("utils/stopwords.txt", "r") as f:
        stopwords = f.readlines()
        stopwords = [converter.convert(word.strip()) for word in stopwords]

    print(*stopwords, sep="\n")