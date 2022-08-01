import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class BasicModel(nn.Module):
    def __init__(self, inp_voc, out_voc, emb_size=64, hid_size=128):
        """
        A simple encoder-decoder seq2seq model
        """
        super().__init__()  # constructor of the parent
                            # initialize base class to track sub-layers, parameters, etc.

        self.inp_voc, self.out_voc = inp_voc, out_voc
        self.hid_size = hid_size  # size of vectors h_t0, h_t1, ...
        
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size) # creates input embedding lookup table (len(inp_voc) X emb_size) of random generated embeddings 
                                                            # to get an embedding, use an index in this array                                                    
        self.emb_out = nn.Embedding(len(out_voc), emb_size) # output embedding lookup table
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True)

        self.dec_start = nn.Linear(hid_size, hid_size)
        self.dec0 = nn.GRUCell(emb_size, hid_size)
        self.logits = nn.Linear(hid_size, len(out_voc))
        
    def forward(self, inp, out):
        """ Apply model in training mode """
        initial_state = self.encode(inp)
        return self.decode(initial_state, out)


    def encode(self, inp, **flags):
        """
        :Takes batch of input sequences, computes initial decoder state 
        :param inp: matrix of input tokens [batch, time]
        :returns: initial decoder state tensors, one or many
        """
        inp_emb = self.emb_inp(inp) # get embedding of input sequence from input lookup table 
        batch_size = inp.shape[0]

        enc_seq, [last_state_but_not_really] = self.enc0(inp_emb)  # output_data, h_n_data = my_gru(input_data, h_0_data)
        # enc_seq: [batch, time, hid_size], last_state: [batch, hid_size]
        # enc_seq -> contains the output features (h_t) from the last layer of the GRU, for each t
        # last_state -> last state h_t of encoder (h_0 for decoder)
        
        # note: last_state is not _really_ last because of padding, let's find the real last_state
        lengths = (inp != self.inp_voc.eos_ix).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]
        # ^-- shape: [batch_size, hid_size]
        
        dec_start = self.dec_start(last_state)
        return [dec_start] # returns h0_0, h0_1, h0_2, ..., h0_n for decoder 
                           # indices 0,1,2,...,n correspond to number of samples in batch

    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state (h_0) and tokens, returns new state and logits for next token
        :param prev_state: a list of previous decoder state tensors, same as returned by encode(...)
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, len(out_voc)]
        """
        prev_gru0_state = prev_state[0]

        # 1. calculate embeddings for prev_tokens
        prev_emb = self.emb_out(prev_tokens)
        # 2. find h_(t+1) for each sample in batch [5 x 128]
        new_dec_state = self.dec0(prev_emb, prev_gru0_state)
        # 3. calculate the output logits
        output_logits = self.logits(new_dec_state)
        
        return [new_dec_state], output_logits

    def decode(self, initial_state, out_tokens, **flags):
        """ Iterate over reference tokens (out_tokens) with decode_step """
        batch_size = out_tokens.shape[0]
        state = initial_state
        
        # initial logits: always predict BOS
        onehot_bos = F.one_hot(torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64),
                               num_classes=len(self.out_voc)).to(device=out_tokens.device)
        first_logits = torch.log(onehot_bos.to(torch.float32) + 1e-9)
        
        logits_sequence = [first_logits]
        for i in range(out_tokens.shape[1] - 1):
            state, logits = self.decode_step(state, out_tokens[:, i])
            logits_sequence.append(logits)
        return torch.stack(logits_sequence, dim=1)

    def decode_inference(self, initial_state, max_len=100, **flags):
        """ Generate translations from model (greedy version) == (beam_search with beam_size=1)"""
        batch_size, device = len(initial_state[0]), initial_state[0].device
        state = initial_state
        # generate first output after BOS
        outputs = [torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64, 
                              device=device)]
        all_states = [initial_state]

        for i in range(max_len):
            state, logits = self.decode_step(state, outputs[-1])
            outputs.append(logits.argmax(dim=-1))
            all_states.append(state)
        
        return torch.stack(outputs, dim=1), all_states

    def decode_inference_beam_search(self, initial_state, beam_size=2, max_len=100, **flags):
        """ Generate beam_size translations(hypos) from model """
        batch_size, device = len(initial_state[0]), initial_state[0].device
        state = initial_state

        outputs = [torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64, 
                              device=device)] # outputs for first BOS
        probs = np.zeros(shape=(beam_size, batch_size))
        states = [deepcopy(initial_state[0].detach()) for _ in range(beam_size)]  # [beam_size, [batch_size, hid_size]]

        for _ in range(max_len):
            next_beams = [[] for _ in range(batch_size)]
            states_history = []
            # beam loop:
            for i in range(len(outputs)): # at the beginning len(outputs) == 1 (outputs = [[bos_ix, bos_ix, ... , bos_ix]])
                prev_tokens = torch.tensor([token for token in outputs[i]], device=device)
                cur_states, logits = self.decode_step(states[i], prev_tokens)
                log_probs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy() # log-probabilities [batch_size, out_len]
                states_history.append(cur_states)
                # finish this code
                

    def translate_lines(self, inp_lines, device, beam_size=None, **kwargs):
        inp = self.inp_voc.to_matrix(inp_lines).to(device)
        initial_state = self.encode(inp)
        if beam_size == None:
            out_ids, states = self.decode_inference(initial_state, **kwargs)
        else:
            out_ids, states = self.decode_inference_beam_search(initial_state, beam_size, **kwargs), None
        return self.out_voc.to_lines(out_ids.cpu().numpy()), states


class AttentionLayer(nn.Module):
    def __init__(self, enc_size, dec_size, hid_size, activ=torch.tanh):
        """ A layer that computes additive attention response and weights """
        super().__init__()

        self.enc_size = enc_size # num units in encoder state
        self.dec_size = dec_size # num units in decoder state
        self.hid_size = hid_size # attention layer hidden units
        self.activ = activ       # attention layer hidden nonlinearity
        
        self.linear_enc = nn.Linear(enc_size, hid_size)
        self.linear_dec = nn.Linear(dec_size, hid_size)
        self.linear_out = nn.Linear(hid_size, 1)
        

    def forward(self, enc, dec, inp_mask):
        """
        Computes attention response and weights
        :param enc: encoder activation sequence, float32[batch_size, ninp, enc_size]
        :param dec: single decoder state used as "query", float32[batch_size, dec_size]
        :param inp_mask: mask on enc activatons (0 after first eos), float32 [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
            - attn - attention response vector (weighted sum of enc)
            - probs - attention weights after softmax
        """

        # Compute logits (logits = [a0,...,a_t,..., a_T] - float32 values)
        # a_t = Linear_out(tanh(Linear_enc(h_enc_t) + Linear_dec(h_dec)))
        logits = self.linear_out(self.activ( self.linear_enc(enc) + self.linear_dec(dec) ))

        # Apply mask - if mask is 0, logits should be -inf or -1e9
        # You may need torch.where
        # masked_logits = logits[inp_mask] 
        masked_logits = torch.where(inp_mask, logits, -1e9)

        # Compute attention probabilities (softmax)
        # p_t = softmax(a_t, [a_0, ..., a_T])
        probs = nn.softmax(logits)

        # Compute attention response using enc and probs
        attn = probs * enc

        return attn, probs


class AttentiveModel(BasicModel):
    def __init__(self, name, inp_voc, out_voc,
                 emb_size=64, hid_size=128, attn_size=128, bid=False):
        """ Translation model that uses attention. """
        super().__init__(inp_voc, out_voc, emb_size, hid_size)
        
        # We could've used these 3 functions below from BasicModel, but here we add bidirectionality
        self.enc0 = nn.GRU(emb_size, hid_size, batch_first=True, bidirectional=bid)
        self.dec_start = nn.Linear(hid_size + hid_size * bid, hid_size)
        self.dec0 = nn.GRUCell(emb_size + hid_size + hid_size * bid, hid_size)
        self.attn = AttentionLayer(hid_size + hid_size * bid, hid_size, attn_size)


    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """

        # encode input sequence, create initial decoder states
        inp_emb = self.emb_inp(inp)

        enc_seq, [last_state_but_not_really] = self.enc0(inp_emb) 
        # enc_seq: [batch, time, hid_size], last_state: [batch, hid_size]
        
        lengths = (inp != self.inp_voc.eos_ix).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = enc_seq[torch.arange(len(enc_seq)), lengths]
        # ^-- shape: [batch_size, hid_size]
        
        dec_start = self.dec_start(last_state) # initial state for decoder

        # compute mask for input sequence
        enc_mask = self.out_voc.compute_mask(inp)
        
        # apply attention layer from initial decoder hidden state 
        _, first_attn_probas = self.attn(enc_seq, dec_start, enc_mask)
        
        return [dec_start, enc_seq, enc_mask, first_attn_probas]
   

    def decode_step(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, n_tokens]
        """
        
        prev_gru0_state, enc_seq, enc_mask, first_attn_probas = prev_state
        attn, attn_probs = self.attn(enc_seq, prev_gru0_state, enc_mask)

        x = self.emb_out(prev_tokens)
        
        x = torch.cat([attn, x], dim=-1)
        x = self.dec0(x, prev_gru0_state)
        
        new_dec_state = [x, enc_seq, enc_mask, attn_probs]
        output_logits = self.logits(x)
        
        return new_dec_state, output_logits




