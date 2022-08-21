To-do:

1. __Figure out why ``beam_search`` function doesn't work with beam_size = 1 (???)__
2. __Fix long translations in beam_search__ 
make sure the code below works fine (gives BLEU > 16.5)
```python 
compute_bleu(model, inp, out, beam_size=beam_size, device=device) 
```
3. Visualise attentions
4. Refactor ``class AttentiveModel`` (add param ``attn_method='Luong'`` or ``attn_method='Bagdanau'``)