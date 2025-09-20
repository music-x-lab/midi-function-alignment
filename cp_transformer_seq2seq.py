import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from cp_transformer_fine_tune import RoFormerSymbolicTransformerInjected, RoformerFineTune, PreprocessingParameters, train
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, fill_with_neg_inf
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import sys
from generator_helper import end_generator
from modules.hubert import _compute_mask
from transformers.models.roformer.modeling_roformer import RoFormerModel, RoFormerConfig, RoFormerEncoder
from cp_transformer import CPTokenizer

MAX_STEPS = 200000
class RoformerEncoderDecoder(L.LightningModule):

    def __init__(self, max_position_embeddings=1536, with_velocity=False):
        super().__init__()
        self.hidden_size = 256
        self.num_layers = 3
        self.num_attention_heads = 4
        self.intermediate_size = 512
        self.local_model_num_layers = 2
        self.local_model_num_attention_heads = 4
        self.local_model_intermediate_size = 512
        encoder_roformer_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=max_position_embeddings,
        )
        main_roformer_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=max_position_embeddings,
            is_decoder=True,
            add_cross_attention=True,
            cross_attention_hidden_size=self.hidden_size,
        )
        self.encoder = RoFormerEncoder(encoder_roformer_config)
        self.model = self.get_base_model(main_roformer_config)
        local_encoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        local_decoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=True,
        )
        self.tokenizer = CPTokenizer(with_velocity=with_velocity)
        self.local_embedding = nn.Embedding(self.tokenizer.n_tokens, self.hidden_size)
        self.local_encoder = RoFormerEncoder(local_encoder_config)
        self.local_decoder = RoFormerEncoder(local_decoder_config)
        self.final_decoder = nn.Linear(self.hidden_size, self.tokenizer.n_tokens)
        self.global_sos = nn.Parameter(torch.randn(self.hidden_size))
        self._future_mask = torch.empty(0)
        self.with_velocity = with_velocity

    def get_base_model(self, config):
        return RoFormerEncoder(config)

    def local_encode(self, x):
        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)
        # prepend a <sos> token
        x = torch.cat([torch.full((x.shape[0], 1), self.tokenizer.sos_token, dtype=torch.long, device=x.device), x], dim=-1)
        mask = x != self.tokenizer.pad_token
        emb = self.local_embedding(x)
        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]
        # get representation of the first token
        return h[:, 0], emb[:, :-1]

    def local_decode(self, h, emb):
        batch_size, subseq_len, _ = emb.shape
        # Add h as the first token of emb
        h = h.view(batch_size, 1, -1)
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        # Create an autoregressive mask

        h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
        return self.final_decoder(h)

    def local_sampling(self, h, max_subseq_len=32, temperature=1.0, global_step=None, sampling_func=None):
        batch_size, _ = h.shape
        y = torch.zeros((batch_size, 0), dtype=torch.long, device=h.device)
        emb = h[:, None, :]
        eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=h.device)
        past_key_values = None
        local_emb = emb
        for i in range(max_subseq_len):
            h, past_key_values = self.local_decoder(local_emb, past_key_values=past_key_values, use_cache=True, return_dict=False)
            # h_out_ref = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
            # assert torch.allclose(h[:, -1:], h_out_ref[:, -1:], rtol=1e-3, atol=1e-5)
            p = self.final_decoder(h[:, -1])
            if sampling_func is not None:
                p = sampling_func(global_step, i, p)
            if temperature == 0:
                p = F.one_hot(p.argmax(dim=-1), self.tokenizer.n_tokens).float()
            else:
                p = F.softmax(p / temperature, dim=-1)
            y_next = torch.multinomial(p, 1)
            y_next[eos_triggered, :] = self.tokenizer.pad_token  # If EOS has been triggered, pad the rest
            eos_triggered = eos_triggered | (y_next.squeeze(1) == self.tokenizer.eos_token)
            y = torch.cat([y, y_next], dim=1)
            if torch.all(eos_triggered):
                y = torch.cat([y, torch.full((batch_size, max_subseq_len - i - 1), self.tokenizer.pad_token,
                                             dtype=torch.long, device=h.device)], dim=1)
                break
            local_emb = self.local_embedding(y_next)
            # emb = torch.cat([emb, self.local_embedding(y_next)], dim=1)
        return y


    def global_sampling(self, x1, x2, max_seq_len=384, temperature=1.0, sampling_func=None):
        batch_size, seq_len1, subseq_len = x1.shape
        h1, _ = self.local_encode(x1)
        h2, emb2 = self.local_encode(x2)
        h1 = h1.view(batch_size, seq_len1, -1)
        # User encoder to encode x1
        h_encoder = self.encoder(h1).last_hidden_state
        # import time
        # start_time = time.time()
        batch_size, seq_len, subseq_len = x2.shape
        h2, _ = self.local_encode(x2)
        h2 = h2.view(batch_size, seq_len, self.hidden_size)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h2 = torch.cat([sos, h2], dim=1)
        y = [x2[:, i, :] for i in range(seq_len)]  # y will be returned by a list
        past_key_values = None
        h_next = h2
        for i in range(seq_len, max_seq_len):
            if i % 10 == 0:
                print('Sampling', i, '/', max_seq_len)
                # print('Time passed', time.time() - start_time)
            attention_mask = self.buffered_future_mask(h2) if past_key_values is None else None
            h_out, past_key_values = self.model(h_next, attention_mask=attention_mask, past_key_values=past_key_values, encoder_hidden_states=h_encoder, use_cache=True, return_dict=False)
            y_next = self.local_sampling(h_out[:, -1], temperature=temperature, global_step=i, sampling_func=sampling_func)
            y.append(y_next)
            h_next = self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
        return y

    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward(self, x1, x2):
        # Use local encoder to encode subsequences
        batch_size, seq_len1, subseq_len = x1.shape
        _, seq_len2, _ = x2.shape
        h1, emb1 = self.local_encode(x1)
        h2, emb2 = self.local_encode(x2)
        h1 = h1.view(batch_size, seq_len1, -1)
        h2 = h2.view(batch_size, seq_len2, -1)
        # Prepend SOS token and remove the last token
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h2 = torch.cat([sos, h2[:, :-1]], dim=1)
        # User encoder to encode x1
        h_encoder = self.encoder(h1).last_hidden_state
        # Use global transformer to decode x2
        h2 = self.model(h2, attention_mask=self.buffered_future_mask(h2), encoder_hidden_states=h_encoder)[0]
        return self.local_decode(h2, emb2)

    def preprocess(self, x, pitch_shift, tuple_size=4):
        batch_size, seq_length, subseq_length = x.shape
        x = x.long().view(batch_size, seq_length, subseq_length // tuple_size, tuple_size)
        x_processed = torch.zeros(batch_size, seq_length, subseq_length // tuple_size, 2, dtype=torch.long, device=x.device)
        pad_indices = x[:, :, :, 1] == 255
        eos_indices = x[:, :, :, 0] == 254
        is_not_drum = x[:, :, :, 0] != 127
        if self.with_velocity:
            x_processed[:, :, :, 0] = x[:, :, :, 0] + 128 * (x[:, :, :, 3] // 8)  # there are 128 instruments
            x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2] + 16) * 128 + pitch_shift[:, None, None] * is_not_drum
        else:
            x_processed[:, :, :, 0] = x[:, :, :, 0]
            x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2] + 1) * 128 + pitch_shift[:, None, None] * is_not_drum
        x_processed[pad_indices] = self.tokenizer.pad_token
        x_processed[:, :, :, 0][eos_indices] = self.tokenizer.eos_token
        return x_processed.view(batch_size, seq_length, subseq_length // tuple_size * 2)

class RoformerSeq2Seq(RoformerFineTune):

    def __init__(self, train_task, mask_prob=0.25, mask_length=10, compress_ratio_l=1, compress_ratio_r=1, lr=1e-4):
        super().__init__(compress_ratio_l=compress_ratio_l, compress_ratio_r=compress_ratio_r, lr=lr)
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.wrapped_model = RoformerEncoderDecoder()
        self.preprocess_args = PreprocessingParameters(train_task)

    def forward(self, x1, x2, indices1, indices2):
        return self.wrapped_model(x1, x2)

    def loss(self, x, pitch_shift):
        x1, x2, indices1, indices2 = self.preprocess(x, pitch_shift, preprocess_args=self.preprocess_args)
        y = self(x1, x2, indices1, indices2)
        return F.cross_entropy(y.view(-1, self.wrapped_model.tokenizer.n_tokens), x2.reshape(-1), ignore_index=self.wrapped_model.tokenizer.pad_token)

    def global_sampling(self, x1, x2, temperature=1.0, sampling_func=None):
        return self.wrapped_model.global_sampling(x1, x2, temperature=temperature, sampling_func=sampling_func)

def main():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int)
    args.add_argument('--dataset_name', type=str)
    args.add_argument('--train_task', type=str, default=None)
    args.add_argument('--weights_path', type=str, default=None)
    args.add_argument('--mask_prob', type=float, default=0.25)
    args.add_argument('--mask_length', type=int, default=10)
    args.add_argument('--compress_ratio_l', type=int, default=1)
    args.add_argument('--compress_ratio_r', type=int, default=1)
    args.add_argument('--train_length', type=int, default=384)
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--early_stopping_patience', type=int, default=MAX_STEPS)
    args = args.parse_args()
    sample_step = max(args.compress_ratio_l, args.compress_ratio_r)
    train_length = args.train_length
    n_gpus = max(torch.cuda.device_count(), 1)
    train_task = args.dataset_name if args.train_task is None else args.train_task
    model_name = f'cp_transformer_seq2seq_v0.1_batch_{args.batch_size * n_gpus}_{train_task}_mask{args.mask_prob}-{args.mask_length}-step{sample_step}'
    if args.weights_path is not None:
        net = RoformerSeq2Seq.load_from_checkpoint(args.weights_path, strict=False, lr=args.lr)
        print('Loaded from', args.weights_path)
    else:
        net = RoformerSeq2Seq(train_task=train_task, mask_prob=args.mask_prob,
                                mask_length=args.mask_length,
                                compress_ratio_l=args.compress_ratio_l, compress_ratio_r=args.compress_ratio_r,
                                lr=args.lr)
    train_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='train'), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='val'), batch_size=None, num_workers=1, persistent_workers=True)
    train(net, model_name, args.early_stopping_patience, MAX_STEPS, train_set_loader, val_set_loader)

if __name__ == '__main__':
    main()