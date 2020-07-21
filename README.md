# tbrain20_AML
This repository holds the code for 玉山人工智慧公開挑戰賽2020夏季賽.

## Dependencies

### pip
```bash
cat requirements.txt | xargs -n 1 pip install
```

### tensorboard
```bash
pip install tensorboard==1.14.0
pip uninstall tensorboard-plugin-wit
```

### CKIP Tagger
```
pip install ckiptagger
wget http://ckip.iis.sinica.edu.tw/data/ckiptagger/data.zip
unzip data.zip
mv data ckip
```

### SpaCy
```bash
python -m spacy download zh_core_web_lg
```

### Stanza
```python
import stanza
stanza.download("zh")
```

### FastText
- src/plot_keywords.py
```bash
git clone https://github.com/facebookresearch/fastText.git
cd fastText
sudo pip install .
```

## TODO:
- Data Augmentation
    - [x] 將沒有名字的句子去掉
    - [ ] 換名字
    - [ ] 找名單
    - [ ] 拿 Label 來爬文
- Data Analysis
    - [ ] Find keywords with TF-IDF
    - [ ] Check news domain
- Tokenizer
    - [x] ZH Character + EN Word
- Modeling
    - [x] Baseline: BiLSTM + Linear

- [x] Scorer

## Results

| Model | Averaged F-1 | Total Score | Loss |
| - | - | - |
| CKIP NER (all)      | 0.20222 | 1015.78 | - |
| Baseline            | 0.93935 | 462.16  | 0.0000141889 |
| Baseline + Filtered | 0.94482 | 464.85  | 0.0000153003 |
| Baseline + Filtered + NER | 0.96005 | 472.34 | 0.0000153003 |

## Dataset

- There are 5023 rows of data
    - 4651 are without labels
    - The document with most names has 15
- There are 830 unique names (labels).
- Max document length is `5538`.
- The dataset is in `.csv` format `UTF-8` encoding with the below categories:

| 欄位 | 資料說明 |
| - | - |
| news_ID | 新聞流水編 |
| hyperlink | 新聞連結 |
| content | 新聞內文 格式為”首句### 省略內文 ###末句” |
| name | 該篇新聞焦點人物清單，若沒有則為 `[]` |

## Evaluation Metric

<!-- $$
Score = \sum_{i=1}^N f(news_i) \\
f(news_i) = \begin{cases}\begin{aligned}
    1&, \qquad \text{if y} = \varnothing \text{ and p } = \varnothing \\
    0&, \qquad \text{if y} = \varnothing \text{ and p } \neq \varnothing \\
    0&, \qquad \text{if y} \neq \varnothing \text{ and p } = \varnothing \\
    \text{F1}(\text{y,p})&, \qquad \text{if y} = \{n_1,...,n_k\} \text{ and p } = \{n_1,...,n_k\} \\
\end{aligned}\end{cases}
$$ -->

![](https://i.imgur.com/elAnnWZ.png)

<!-- $$
\begin{aligned}
\text{F1} &= \frac{2}{\text{recall}^{-1} + \text{precision}^{-1}} \\
\text{recall} &= \frac{| \ T \ \cap \ P \ |}{T} \\
\text{precision} &= \frac{| \ T \ \cap \ P \ |}{P}
\end{aligned}
$$ -->

![](https://i.imgur.com/sRYktgv.png)

## Documentations

玉山銀行 API 相關文件
- [API 開發說明文件](https://hackmd.io/@UcQg6jwlT_WL_ZNkPZMm6Q/BJfELe_c8)  
- [API 規格說明文件](https://hackmd.io/@nqf_7suCTA2B-tYY2TvmYw/r11xDuMoL)
