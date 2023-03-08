from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import torch


def save_checkpoint(state, filename="saved_weights.pth"):
    # Holds the torch.Tensor objects of all the layers of the model
    # without saving the whole model architecture
    print(f"=> Saving checkpoint at epoch {state['epoch']}")
    torch.save(state,filename)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def compute_loss(model, inp, out, **flags):
    """
    Compute loss (float32 scalar) as in the formula above
    :param inp: input tokens matrix, int32[batch, time]
    :param out: reference tokens matrix, int32[batch, time]
    
    In order to pass the tests, your function should
    * include loss at first EOS but not the subsequent ones
    * divide sum of losses by a sum of input lengths (use voc.compute_mask)
    """
    mask = model.out_voc.compute_mask(out) # [batch_size, out_len]
    targets_1hot = F.one_hot(out, len(model.out_voc)).to(torch.float32)
    
    # outputs of the model, [batch_size, out_len, num_tokens]
    logits_seq = model(inp, out)
    # now compute -log(p(y_t | y_1,..,y_t-1, X, 0))
    # log-probabilities of all tokens at all steps, [batch_size, out_len, num_tokens] 
    # Note: probabilities = softmax(logits)
    logprobs_seq = torch.log_softmax(logits_seq, dim=-1)

    # log-probabilities of correct outputs, [batch_size, out_len]
    logp_out = (logprobs_seq * targets_1hot).sum(dim=-1)  
    # ^-- this will select the probability of the actual next token.
    
    return -logp_out[mask].mean()


def compute_bleu(model, inp_lines, out_lines, device, bpe_sep='@@ ', batch_size=32, **flags):
    """
    Estimates corpora-level BLEU score of model's translations given inp and reference out
    Note: if you're serious about reporting your results, use https://pypi.org/project/sacrebleu
    """
    with torch.no_grad():
        # translations, _ = model.translate_lines(inp_lines, beam_size=2, device=device, **flags) 
        # optimizing memory
        translations = []
        for i in range(0, len(inp_lines), batch_size):
            translations.extend(model.translate_lines(inp_lines[i:i+batch_size], device, **flags)[0])

        translations = [line.replace(bpe_sep, '') for line in translations]
        actual = [line.replace(bpe_sep, '') for line in out_lines]
        return corpus_bleu(
            [[ref.split()] for ref in actual],
            [trans.split() for trans in translations],
            smoothing_function=lambda precisions, **kw: [p + 1.0 / p.denominator for p in precisions]
            ) * 100