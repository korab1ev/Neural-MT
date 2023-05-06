# Files:
1) ``notebook.ipynb`` - Старые эксперименты (2022.07) 
 + **Basic model** (RNN encoder + RNN decoder)
 + **Attentive model** (3-layer Bi-LSTM encoder + GRU decoder with attention)
 + **Larger attentive model** (**emb_size**=256, **hid_size**=256, **attn_size**=256)
 + **Transformer model** (fine-tuning)
2) ``transformer-and-augmentation.ipynb`` - Последние эксперименты (2023)
+ Transformer no-fine-tuning translations
+ Transformer + fine-tuning translations
+ Transformer + fine-tuning + augmentation
3) ``augment-data-prep.ipynb`` - Подготовка данных **GeoNames** для аугментации
4) ``models.py`` - code for models (2022.07)
5) ``utils.py`` - helper functions 

# Articles:

- [On The Evaluation of Machine Translation Systems Trained With Back-Translation](https://arxiv.org/abs/1908.05204.pdf)
- [Domain, Translationese and Noise in Synthetic Data for Neural Machine Translation](https://arxiv.org/abs/1911.03362)
- [Understanding Back-Translation at Scale](https://arxiv.org/abs/1808.09381)
- [Very Deep Transformers for Neural Machine Translation](https://arxiv.org/abs/2008.07772)
- [Simple and Effective Noisy Channel Modeling for Neural Machine Translation](https://arxiv.org/abs/1908.05731.pdf)