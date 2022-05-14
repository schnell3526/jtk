# jtk
This is a tokenizer that uses juman++, which can be used with hugginface transformers, for text preprocessing.

## install
```shell
pip install git+https://github.com/schnell3526/jtk
```

## how to use


```python
from transformers import AutoModelForMaskedLM
from jtk import RobertaJumanTokenizer

tokenizer = RobertaJumanTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")
model = AutoModelForMaskedLM.from_pretrained("nlp-waseda/roberta-base-japanese")

# input should be segmented into words by Juman++ in advance
sentence = '早稲田大学で自然言語処理を[MASK]する。'
encoding = tokenizer(sentence, return_tensors='pt')
```