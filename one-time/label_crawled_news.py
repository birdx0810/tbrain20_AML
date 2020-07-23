import pandas as pd

import ast
import random

"""
script for preprocessing crawled news.
"""


def label_names(
        original_csv_path='../data/tbrain_train_final_0610.csv',
        crawled_csv_path='../bert/other_news.csv',
        save_csv_path='./other_news-v2.csv'):
    """
    Label names in crawled news based on names in tbrain data.
    """

    tbrain_news_df = pd.read_csv(original_csv_path)
    crawled_news_df = pd.read_csv(crawled_csv_path)

    # get names in tbrain news
    name_list = []
    for index, row in tbrain_news_df.iterrows():
        cur_name_list = ast.literal_eval(row["name"])
        name_list.extend(cur_name_list)
    name_list = list(set(name_list))
    print(len(name_list))

    # identify names in crawled news
    labels = []
    for index, row in crawled_news_df.iterrows():
        cur_labels = []
        cur_content = row['content']
        if not isinstance(cur_content, float):
            for name in name_list:
                if name in cur_content:
                    cur_labels.append(name)
        labels.append(cur_labels)

    assert len(labels) == crawled_news_df.shape[0]

    # write crawled news to file with new labels
    crawled_news_df['name'] = labels
    crawled_news_df.to_csv(save_csv_path, index=False)
    return


def filter_same_news(
        crawled_csv_path='./other_news-v2.csv',
        save_csv_path='./other_news-v3.csv'):
    """
    filter duplicate news by urls
    """
    crawled_news_df = pd.read_csv(crawled_csv_path)
    crawled_news_df = crawled_news_df.reset_index()
    print(crawled_news_df.shape)

    ids_to_remove = []
    for name in crawled_news_df.person.unique():
        cur_urls = []
        person_df = crawled_news_df.loc[crawled_news_df['person'] == name]
        for index, row in person_df.iterrows():
            cur_url = row['hyperlink']
            if cur_url not in cur_urls:
                cur_urls.append(cur_url)
            else:
                ids_to_remove.append(row['other_news_ID'])
                # ids_to_remove.append(row.index)

    crawled_news_df.drop(crawled_news_df[crawled_news_df['other_news_ID'].isin(ids_to_remove)].index, inplace=True)
    crawled_news_df.drop('index', axis='columns', inplace=True)
    crawled_news_df['other_news_ID'] = range(1, crawled_news_df.shape[0]+1)
    print(crawled_news_df.shape)
    crawled_news_df.to_csv(save_csv_path, index=False)
    return


def replace_names_from_csv(
        names_list_path='./fake_names_1000.txt',
        news_csv_path='./other_news-v3.csv',
        save_csv_path='./other_news-v4.csv',
        name_format='comma-only',
        num_len_2_limit=35,
        generate_n_data_from_content=5):
    """
    Replace names in contents with fake names for data augmentation.
    """
    with open(names_list_path, 'r') as f:
        fake_names = f.readline().strip('\n').split(', ')

    # make up names with length 2
    num_len_2_count = 0
    for index, name in enumerate(fake_names):
        if len(name) == 2:
            num_len_2_count += 1
        elif len(name) == 3:
            fake_names[index] = name[:2]
            num_len_2_count += 1
        
        if num_len_2_count > num_len_2_limit:
            break

    random.shuffle(fake_names)

    # data augmentation
    if name_format == 'comma-only':
        news_df = pd.read_csv(news_csv_path, keep_default_na=False)
    else:
        news_df = pd.read_csv(news_csv_path)
    print(news_df.shape)

    contents = []
    labels = []
    cur_fake_name_index = 0
    for index, row in news_df.iterrows():
        cur_content = row['content']

        if name_format == 'comma-only':
            # for reading names in format `name1, name2`
            cur_labels_lst = row["name"].split(',')
            if len(cur_labels_lst) == 1 and cur_labels_lst[0] == '':
                continue
            
        else:
            # for reading names in format `["name1", "name2"]`
            cur_labels_lst = ast.literal_eval(row["name"])
        
        for n in range(generate_n_data_from_content):
            cur_fake_names = []
            generated_content = cur_content

            for i in range(len(cur_labels_lst)):
                cur_fake_names.append(fake_names[cur_fake_name_index])
                # replace with fake names
                generated_content = generated_content.replace(cur_labels_lst[i], cur_fake_names[i])
                # update fake name index
                cur_fake_name_index = 0 if cur_fake_name_index + 1 == len(fake_names) else cur_fake_name_index + 1
            
            contents.append(generated_content)
            labels.append(cur_fake_names)

    assert len(contents) == len(labels)

    generated_df = pd.DataFrame({
        'content': contents,
        'name': labels,
    })
    generated_df.to_csv(save_csv_path, index=False)
    print(generated_df.shape)
    
    return


if __name__ == '__main__':
    # label_names()
    # filter_same_news()
    replace_names_from_csv(
        names_list_path='./fake_names_1000.txt',
        news_csv_path='./train.csv',
        save_csv_path='./other_news-v4_aug-from-tbrain.csv')
