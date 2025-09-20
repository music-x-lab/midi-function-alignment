import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, fill_with_neg_inf
from cp_transformer_fine_tune import PreprocessingParameters, shrink_duration, RoFormerSymbolicTransformerInjected, RoformerFineTune, train
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from peft import LoraConfig, get_peft_model
from modules.hubert import _compute_mask
from generator_helper import end_generator
from yield_tags import Tags

MAX_STEPS = 60000

class RoformerPrompt(RoformerFineTune):

    def __init__(self, model_fp, train_task, mask_prob=None, mask_length=None, max_position_embeddings=768,
                 compress_ratio_l=1, compress_ratio_r=1, lr=None, change_pe=True):
        super().__init__(compress_ratio_l=compress_ratio_l, compress_ratio_r=compress_ratio_r, lr=lr)
        self.save_hyperparameters()
        if not os.path.exists(model_fp):
            # Use relative path
            model_fp = os.path.join('ckpt', os.path.basename(model_fp))
        base_model = RoFormerSymbolicTransformerInjected.load_from_checkpoint(model_fp, max_position_embeddings=max_position_embeddings, strict=False)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["query", "value"],
            lora_dropout=0.1,
        )
        self.wrapped_model = get_peft_model(base_model, lora_config)
        self.n_layers = base_model.num_layers
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.change_pe = change_pe
        self.preprocess_args = PreprocessingParameters(train_task)
        self.masked_embedding = nn.Parameter(torch.randn(1, 1, base_model.hidden_size) * 0.1, requires_grad=True)
        if self.change_pe:
            self.sentence_embedding = nn.Parameter(torch.zeros(2, base_model.hidden_size // base_model.num_attention_heads), requires_grad=True)

    def forward(self, x1, x2, indices1, indices2):
        batch_size, seq_len1, subseq_len = x1.shape
        _, seq_len2, _ = x2.shape
        concat_input = torch.cat([x1, x2], dim=1)
        gen = self.wrapped_model(concat_input)
        data = next(gen)
        assert data[0] == Tags.SIMUNOTE_EMBEDDING
        emb = data[1]
        emb1 = emb[:, :seq_len1, :]
        emb2 = emb[:, seq_len1:, :]
        if self.mask_prob > 0.0:
            if self.preprocess_args.left_mask:
                mask1 = _compute_mask((x1.size(0), x1.size(1)), self.mask_prob, self.mask_length, x1.device, 2)
                emb1[mask1] = self.masked_embedding.to(emb1.dtype)
            elif self.training:
                mask2 = _compute_mask((x2.size(0), x2.size(1)), self.mask_prob, self.mask_length, x2.device, 2)
                emb2[mask2] = self.masked_embedding.to(emb2.dtype)
            data[1] = torch.cat([emb1, emb2], dim=1)
        data = next(gen)
        assert data[0] == Tags.PE_POSITIONS
        if self.change_pe:
            pe = data[1]
            sentence_embedding = self.sentence_embedding[(pe >= seq_len1).long()]
            pe[seq_len1:] -= seq_len1  # the second part restarts from 0
            data[2] = sentence_embedding
        for layer in range(self.n_layers):
            data = next(gen)
            assert data[0] == Tags.HIDDEN_STATES
            data = next(gen)
            assert data[0] == Tags.PRENORM_OUTPUT
        return end_generator(gen)

    def global_sampling(self, x1, x2, temperature=1.0, sampling_func=None):
        print('Prompt-based Sampling')
        batch_size, seq_len1, subseq_len = x1.shape
        _, seq_len2, _ = x2.shape
        max_seq_len = seq_len1 * self.compress_ratio_l // self.compress_ratio_r
        concat_input = torch.cat([x1, x2], dim=1)

        gen = self.wrapped_model.global_sampling(concat_input, max_seq_len=seq_len1 + max_seq_len, temperature=temperature, sampling_func=sampling_func)

        for i in range(seq_len1 + seq_len2, seq_len1 + max_seq_len):
            data = next(gen)
            assert data[0] == Tags.GENERATION_STEP
            data = next(gen)
            assert data[0] == Tags.PE_POSITIONS
            if self.change_pe:
                pe = data[1]
                sentence_embedding = self.sentence_embedding[(pe >= seq_len1).long()]
                pe[pe >= seq_len1] -= seq_len1  # the second part restarts from 0
                data[2] = sentence_embedding
            for layer in range(self.n_layers):
                data = next(gen)
                assert data[0] == Tags.HIDDEN_STATES
                data = next(gen)
                assert data[0] == Tags.PRENORM_OUTPUT
        result = end_generator(gen)
        # Get newly generated tokens
        return result[seq_len1:]  # TODO: the length seems to be 1 less than expected

    def loss(self, x, pitch_shift):
        x1, x2, indices1, indices2 = self.preprocess(x, pitch_shift, preprocess_args=self.preprocess_args)
        batch_size, seq_length1, subseq_length = x1.shape
        _, seq_length2, _ = x2.shape
        y = self(x1, x2, indices1, indices2)
        # only take the second part
        y = y.view(batch_size, seq_length1 + seq_length2, subseq_length, y.shape[-1])[:, seq_length1:, :, :]
        return F.cross_entropy(y.reshape(-1, self.wrapped_model.tokenizer.n_tokens), x2.reshape(-1), ignore_index=self.wrapped_model.tokenizer.pad_token)


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
    args.add_argument('--change_pe', action='store_true', default=True)
    args.add_argument('--no_change_pe', action='store_false', dest='change_pe')
    args = args.parse_args()
    sample_step = max(args.compress_ratio_l, args.compress_ratio_r)
    train_length = args.train_length
    n_gpus = max(torch.cuda.device_count(), 1)
    train_task = args.dataset_name if args.train_task is None else args.train_task
    model_name = 'prompt' if args.change_pe else 'prompt_no_pe'
    model_name = f'cp_transformer_{model_name}_lora_v1.1_batch_{args.batch_size * n_gpus}_{train_task}_mask{args.mask_prob}-{args.mask_length}-step{sample_step}'
    if args.weights_path is not None:
        net = RoformerPrompt.load_from_checkpoint(args.weights_path, strict=False, lr=args.lr)
        print('Loaded from', args.weights_path)
    else:
        net = RoformerPrompt(args.fp_path, train_task=train_task, mask_prob=args.mask_prob,
                             mask_length=args.mask_length,
                             compress_ratio_l=args.compress_ratio_l, compress_ratio_r=args.compress_ratio_r,
                             lr=args.lr, change_pe=args.change_pe)
    train_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='train', sample_step=sample_step), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='val', sample_step=sample_step), batch_size=None, num_workers=1, persistent_workers=True)
    train(net, model_name, args.early_stopping_patience, MAX_STEPS, train_set_loader, val_set_loader)

if __name__ == '__main__':
    main()
    # sanity_check()