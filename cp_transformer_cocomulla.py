import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from cp_transformer_fine_tune import RoFormerSymbolicTransformerInjected, RoformerFineTune, PreprocessingParameters, train
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, fill_with_neg_inf
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
import sys
from generator_helper import end_generator
from modules.hubert import _compute_mask
from yield_tags import Tags

MAX_STEPS = 40000

class RoFormerSymbolicTransformerInjectedCocoMulla(RoFormerSymbolicTransformerInjected):

    def __init__(self, size=1, max_position_embeddings=512):
        super().__init__(size=size, max_position_embeddings=max_position_embeddings)

    def forward_from_simunote(self, h):
        sos = self.global_sos.view(1, 1, -1).repeat(h.shape[0], 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)
        # Use global transformer to decode
        yield from self.model(h, attention_mask=None)  # no causal mask
        raise Exception('This generator should not go beyond the last yield statement')


    def global_sampling(self, x, max_seq_len=384, temperature=1.0, sampling_func=None):
        batch_size, seq_len, subseq_len = x.shape
        h, _ = self.local_encode(x)
        h = h.view(batch_size, seq_len, h.shape[-1])
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list
        past_key_values = None
        for i in range(seq_len, max_seq_len):
            yield [Tags.GENERATION_STEP, i]
            if i % 10 == 0:
                print('Sampling', i, '/', max_seq_len)
            injected_array = [Tags.SIMUNOTE_EMBEDDING, h]
            yield injected_array
            h = injected_array[1]
            attention_mask = self.buffered_future_mask(h) if past_key_values is None else None
            h_out, past_key_values = yield from self.model(h, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=False)
            y_next = self.local_sampling(h_out[:, -1], temperature=temperature, global_step=i, sampling_func=sampling_func)
            y.append(y_next)
            h = self.local_encode(y_next.unsqueeze(1))[0].unsqueeze(1)
        return y
class RoformerCocoMulla(RoformerFineTune):

    def __init__(self, model_fp, train_task, mask_prob=None, mask_length=None, max_position_embeddings=384,
                 compress_ratio_l=1, compress_ratio_r=1, lr=None):
        super().__init__(compress_ratio_l=compress_ratio_l, compress_ratio_r=compress_ratio_r, lr=lr)
        self.save_hyperparameters()
        if not os.path.exists(model_fp):
            # Use relative path
            model_fp = os.path.join('ckpt', os.path.basename(model_fp))
        base_model = RoFormerSymbolicTransformerInjectedCocoMulla.load_from_checkpoint(model_fp, strict=False)
        base_model.eval()
        base_model.freeze()
        self.wrapped_model = base_model
        self.n_layers = base_model.num_layers
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.preprocess_args = PreprocessingParameters(train_task)
        assert compress_ratio_l == 1 and compress_ratio_r == 1  # not supported yet
        # Trainable parameters
        self.h0p = nn.Parameter(torch.randn(1, 384, base_model.hidden_size), requires_grad=True)
        self.z_pos = nn.Parameter(torch.randn(self.n_layers, 1, 384, base_model.hidden_size), requires_grad=True)
        self.gates = nn.Parameter(torch.zeros(self.n_layers), requires_grad=True)
        self.masked_embedding = nn.Parameter(torch.randn(1, 1, base_model.hidden_size), requires_grad=True)
        self.w_e = nn.ModuleList([
            nn.Linear(base_model.hidden_size, base_model.hidden_size) for _ in range(self.n_layers)
        ])

    def forward(self, x1, x2):
        gen1 = self.wrapped_model(x1, use_causal_mask=False)
        gen2 = self.wrapped_model(x2)
        data1 = next(gen1); assert data1[0] == Tags.SIMUNOTE_EMBEDDING
        data2 = next(gen2); assert data2[0] == Tags.SIMUNOTE_EMBEDDING
        z = data1[1]
        data1[1] = self.h0p.expand_as(z)  # replace by learnable embedding, eq. 6
        if self.mask_prob > 0.0:
            if self.preprocess_args.left_mask:
                mask1 = _compute_mask((x1.size(0), x1.size(1)), self.mask_prob, self.mask_length, x1.device, 2)
                data1[1][mask1] = self.masked_embedding.to(data1[1].dtype)
            elif self.training:
                mask2 = _compute_mask((x2.size(0), x2.size(1)), self.mask_prob, self.mask_length, x2.device, 2)
                data2[1][mask2] = self.masked_embedding.to(data2[1].dtype)

        gen3 = self.wrapped_model.forward_from_simunote(data2[1])  # simulates cross-attention

        data1 = next(gen1); assert data1[0] == Tags.PE_POSITIONS
        data2 = next(gen2); assert data2[0] == Tags.PE_POSITIONS
        data3 = next(gen3); assert data3[0] == Tags.PE_POSITIONS
        for layer in range(self.n_layers):
            data1 = next(gen1); assert data1[0] == Tags.HIDDEN_STATES
            data2 = next(gen2); assert data2[0] == Tags.HIDDEN_STATES
            data3 = next(gen3); assert data3[0] == Tags.HIDDEN_STATES
            data1[1] = data1[1] + self.w_e[layer](z + self.z_pos[layer])  # inject z, eq. 8
            data3[1] = data2[1] + data1[1]  # eq. 12, q is from main + prefix
            data3[2] = data1[1]  # eq. 12, k and v are from prefix, use encoder hidden states

            data1 = next(gen1); assert data1[0] == Tags.PRENORM_OUTPUT
            data2 = next(gen2); assert data2[0] == Tags.PRENORM_OUTPUT
            data3 = next(gen3); assert data3[0] == Tags.PRENORM_OUTPUT
            data2[1] = data2[1] + data3[1] * self.gates[layer]  # eq. 13
        return end_generator(gen2)

    def global_sampling(self, x1, x2, temperature=1.0, sampling_func=None):
        print('Cocomulla Sampling')
        batch_size, max_seq_len, subseq_len = x1.shape
        seq_len = x2.shape[1]
        # First process all hidden states from gen1
        gen1 = self.wrapped_model(x1, use_causal_mask=False)
        data1 = next(gen1); assert data1[0] == Tags.SIMUNOTE_EMBEDDING
        z = data1[1]
        data1[1] = self.h0p.expand_as(z)  # replace by learnable embedding, eq. 6
        data1 = next(gen1); assert data1[0] == Tags.PE_POSITIONS
        hidden_states_gen1 = []
        for layer in range(self.n_layers):
            data1 = next(gen1); assert data1[0] == Tags.HIDDEN_STATES
            data1[1] = data1[1] + self.w_e[layer](z + self.z_pos[layer])  # inject z, eq. 8
            hidden_states_gen1.append(data1[1])
            data1 = next(gen1); assert data1[0] == Tags.PRENORM_OUTPUT
        # Now create gen2
        gen2 = self.wrapped_model.global_sampling(x2, max_seq_len=max_seq_len, temperature=temperature, sampling_func=sampling_func)
        for i in range(seq_len, max_seq_len):
            data2 = next(gen2); assert data2[0] == Tags.GENERATION_STEP
            data2 = next(gen2); assert data2[0] == Tags.SIMUNOTE_EMBEDDING
            gen3 = self.wrapped_model.forward_from_simunote(data2[1])  # simulates cross-attention
            data2 = next(gen2); assert data2[0] == Tags.PE_POSITIONS
            data3 = next(gen3); assert data3[0] == Tags.PE_POSITIONS
            for layer in range(self.n_layers):
                data2 = next(gen2); assert data2[0] == Tags.HIDDEN_STATES
                data3 = next(gen3); assert data3[0] == Tags.HIDDEN_STATES
                h_length = data2[1].shape[1]
                data3[1] = data2[1] + data1[1][:, i + 1 - h_length:i + 1]  # eq. 12, q is from main + prefix
                data3[2] = hidden_states_gen1[layer]  # eq. 12, k and v are from prefix, use encoder hidden states

                data2 = next(gen2); assert data2[0] == Tags.PRENORM_OUTPUT
                data3 = next(gen3); assert data3[0] == Tags.PRENORM_OUTPUT
                data2[1] = data2[1] + data3[1] * self.gates[layer]  # eq. 13x
        return end_generator(gen2)

    def loss(self, x, pitch_shift):
        x1, x2, indices1, indices2 = self.preprocess(x, pitch_shift, preprocess_args=self.preprocess_args)
        y = self(x1, x2)
        return F.cross_entropy(y.view(-1, self.wrapped_model.tokenizer.n_tokens), x2.reshape(-1), ignore_index=self.wrapped_model.tokenizer.pad_token)


def main():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int)
    args.add_argument('--fp_path', type=str, default='ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt')
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
    model_name = f'cp_transformer_cocomulla_v3.1_batch_{args.batch_size * n_gpus}_{train_task}_mask{args.mask_prob}-{args.mask_length}-step{sample_step}'
    if args.weights_path is not None:
        net = RoformerCocoMulla.load_from_checkpoint(args.weights_path, strict=False, lr=args.lr)
        print('Loaded from', args.weights_path)
    else:
        net = RoformerCocoMulla(args.fp_path, train_task=train_task, mask_prob=args.mask_prob,
                                mask_length=args.mask_length,
                                compress_ratio_l=args.compress_ratio_l, compress_ratio_r=args.compress_ratio_r,
                                lr=args.lr)
    train_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='train'), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='val'), batch_size=None, num_workers=1, persistent_workers=True)
    train(net, model_name, args.early_stopping_patience, MAX_STEPS, train_set_loader, val_set_loader)

if __name__ == '__main__':
    main()