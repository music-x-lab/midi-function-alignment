import numpy as np

from cp_transformer_yinyang import RoformerYinyang, PreprocessingParameters
from cp_transformer_continuous import RoformerPrompt
from cp_transformer_cocomulla import RoformerCocoMulla
from cp_transformer_seq2seq import RoformerSeq2Seq
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES
from cp_transformer_inference import decode_output, inference_perplexity
from cp_transformer_fake import FakeModel
from settings import RWC_DATASET_PATH
import torch
import pretty_midi
import os
import sys
from settings import *

def decompress(model, byte_arr):
    x = torch.tensor(byte_arr).unsqueeze(0)
    x = x.cuda()
    return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda(), preprocess_args=PreprocessingParameters(''))[:2]

def get_sampling_function(ins=None, rhythm=None, max_polyphony=16):
    def sampling_function(global_pos, local_pos, distribution):
        batch_size, vocab_size = distribution.shape
        n_normal_tokens = 24 * 128 + 256
        sos_token = n_normal_tokens
        eos_token = n_normal_tokens + 1
        pad_token = n_normal_tokens + 2
        assert vocab_size == n_normal_tokens + 3  # 24 durations, 128 pitches, 128 instruments (padded to 256)
        output_mask = torch.ones(vocab_size, device=distribution.device, dtype=torch.bool)  # disable all tokens
        # Aggregation mask calculates how many entries should contribute to the eos probability
        aggregation_mask = torch.zeros(vocab_size, device=distribution.device, dtype=torch.bool)
        aggregation_mask[eos_token] = True
        if rhythm is not None and global_pos % rhythm != 0:  # should output EOS at the first step
            assert local_pos == 0
            output_mask[eos_token] = False
        elif local_pos % 2 == 0:  # this decodes the instrument
            current_polyphony = local_pos // 2
            if current_polyphony >= max_polyphony:
                pass  # do not generate more notes
            elif ins is not None:  # additional instrument mask
                output_mask[ins] = False
                aggregation_mask[:128] = True
                aggregation_mask[ins] = False
            else:
                output_mask[:128] = False  # there are only 128 instruments
            output_mask[eos_token] = False  # can generate EOS
        else:  # this decodes the pitch and duration
            output_mask[256:n_normal_tokens] = False  # should not output instrument
        distribution = distribution.masked_fill(output_mask, -float('inf'))
        '''if output_mask[eos_token]:
            # Aggregate the EOS probability
            distribution[:, eos_token] = distribution[:, aggregation_mask].logsumexp(dim=1)'''
        return distribution
    return sampling_function

def get_unquantized_midi_track(midi_path, ins_ids=None, generation_length=384, fixed_velocity=100, fixed_tempo=120, fixed_program=None):
    results = preprocess_midi(midi_path, 16, ins_ids=ins_ids, filter=False, fixed_length=generation_length, return_midi_ins=True)
    concat_result = []
    if not isinstance(fixed_velocity, list):
        fixed_velocity = [fixed_velocity] * len(ins_ids)
    if not isinstance(fixed_program, list):
        fixed_program = [fixed_program] * len(ins_ids)
    for i, result in enumerate(results):
        scale_ratio = 60 / fixed_tempo / 4
        for ins in result:
            ins.notes = [pretty_midi.Note(start=note.start * scale_ratio, end=note.end * scale_ratio, pitch=note.pitch, velocity=fixed_velocity[i]) for note in ins.notes]
            concat_result.append(ins)
            if fixed_program[i] is not None:
                assert not ins.is_drum
                ins.program = fixed_program[i]
    return concat_result

def cond_continuation(model, midi_path, prompt_length=100, generation_length=384, temperature=1.0, n_samples=1, ins_ids=None, sampling_func=None,
                      output_bpm=150, seed=0, fixed_velocity=100, fixed_program=None):
    if not isinstance(fixed_velocity, list):
        fixed_velocity = [fixed_velocity] * len(ins_ids)
    if not isinstance(fixed_program, list):
        fixed_program = [fixed_program] * len(ins_ids)
    if ins_ids is None:
        raise ValueError('ins_ids must be provided')
    folder = f'prompt{prompt_length}_{model.save_name}'
    ins_ids_str = ','.join(ins_ids)
    file_name_pattern = f'temp/{folder}/{os.path.basename(midi_path)}_temp{temperature}_continuation_%d[{ins_ids_str}].mid'
    if os.path.exists(file_name_pattern % (n_samples - 1)) and:
        print(f'Already exists: {file_name_pattern % (n_samples - 1)}')
        return
    x1, x2 = decompress(model, preprocess_midi(midi_path, 16, ins_ids=ins_ids, filter=False, fixed_length=generation_length)[0])
    print(x1.shape, x2.shape)
    x1 = x1[:, :generation_length]
    ratio1 = model.compress_ratio_l
    ratio2 = model.compress_ratio_r
    if isinstance(model, FakeModel):  # ground-truth model
        decode_output((),
                      f'temp/{folder}/{os.path.basename(midi_path)}_gt[{ins_ids_str}].mid',
                      ratio=(ratio1, ratio2),
                      tempo=output_bpm,
                      extra_instruments=get_unquantized_midi_track(midi_path,
                                                                   ins_ids=ins_ids,
                                                                   generation_length=generation_length,
                                                                   fixed_tempo=output_bpm,
                                                                   fixed_velocity=fixed_velocity,
                                                                   fixed_program=fixed_program))
        return
    else:
        x2 = x2[:, :prompt_length]
        cond_notes = [x1[:, i, :] for i in range(x1.shape[1])]
        decode_output(([x2[:, i, :] for i in range(x2.shape[1])],),
                      f'temp/{folder}/{os.path.basename(midi_path)}_prompt[{ins_ids_str}].mid',
                      ratio=(ratio1, ratio2),
                      tempo=output_bpm,
                      velocity=fixed_velocity[1],
                      fixed_program=fixed_program[1],
                      extra_instruments=get_unquantized_midi_track(midi_path,
                                                                   ins_ids=ins_ids[:1],
                                                                   generation_length=generation_length,
                                                                   fixed_tempo=output_bpm,
                                                                   fixed_velocity=fixed_velocity[0],
                                                                   fixed_program=fixed_program[0]))

    with torch.no_grad():
        x1 = x1.repeat(n_samples, 1, 1)
        x2 = x2.repeat(n_samples, 1, 1)
        torch.random.manual_seed(seed)  # fix the seed for reproducibility
        np.random.seed(seed)
        output = model.global_sampling(x1, x2, temperature=temperature, sampling_func=sampling_func)
    for i in range(n_samples):
        output_i = [output[j][i:i + 1, :] for j in range(len(output))]
        decode_output((output_i,), file_name_pattern % i,
                      ratio=(ratio1, ratio2),
                      tempo=output_bpm,
                      velocity=fixed_velocity[1],
                      fixed_program=fixed_program[1],
                      extra_instruments=get_unquantized_midi_track(midi_path,
                                                                   ins_ids=ins_ids[:1],
                                                                   generation_length=generation_length,
                                                                   fixed_tempo=output_bpm,
                                                                   fixed_velocity=fixed_velocity[0],
                                                                   fixed_program=fixed_program[0]))

def eval_model(model_name):
    if model_name.startswith('ground-truth:'):
        task_name = model_name.split(':')[1]
        model = FakeModel(task_name)
    else:
        model_dir = model_name.split('.epoch')[0]
        if os.path.exists(f'ckpt/{model_name}'):
            model_path = f'ckpt/{model_name}'
        elif os.path.exists(f'ckpt/{model_dir}'):
            model_path = f'ckpt/{model_dir}/{model_name}'
        else:
            raise FileNotFoundError(f'Cannot find model: {model_name}')
        model_type = RoformerCocoMulla if 'cocomulla' in model_path else \
            RoformerPrompt if 'prompt' in model_path else \
            RoformerYinyang if 'yinyang' in model_path else \
            RoformerSeq2Seq if 'seq2seq' in model_path else None
        assert model_type is not None
        model = model_type.load_from_checkpoint(model_path, strict=False)
        model.save_name = os.path.basename(model_path)
    model.cuda()
    model.eval()
    if 'drums_nondrum' in model.save_name:
        if '_rev' in model.save_name:
            cond_continuation(model, 'input/RM-P001.SMF_SYNC_BPM135.MID', fixed_velocity=[80, 100],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=2, ins_ids=['nondrum', 'drum'], output_bpm=135)
            cond_continuation(model, 'input/RM-P003.SMF_SYNC_BPM110.MID', fixed_velocity=[80, 100],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=2, ins_ids=['nondrum', 'drum'], output_bpm=135)
            cond_continuation(model, 'input/RM-P090.SMF_SYNC_BPM127.MID', fixed_velocity=[80, 100],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=2, ins_ids=['nondrum', 'drum'], output_bpm=110)
            cond_continuation(model, 'input/RM-P005.SMF_SYNC_BPM135.MID', fixed_velocity=[80, 100],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=2, ins_ids=['nondrum', 'drum'], output_bpm=135)
            cond_continuation(model, 'input/RM-P008.SMF_SYNC_BPM130.MID', fixed_velocity=[80, 100],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=2, ins_ids=['nondrum', 'drum'], output_bpm=130)
        else:
            cond_continuation(model, 'input/RM-P001.SMF_SYNC_BPM135.MID', fixed_velocity=[100, 80],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=32, ins_ids=['drum', 'nondrum'], output_bpm=135)
            cond_continuation(model, 'input/RM-P003.SMF_SYNC_BPM110.MID', fixed_velocity=[100, 80],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=32, ins_ids=['drum', 'nondrum'], output_bpm=135)
            cond_continuation(model, 'input/RM-P090.SMF_SYNC_BPM127.MID', fixed_velocity=[100, 80],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=32, ins_ids=['drum', 'nondrum'], output_bpm=110)
            cond_continuation(model, 'input/RM-P005.SMF_SYNC_BPM135.MID', fixed_velocity=[100, 80],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=32, ins_ids=['drum', 'nondrum'], output_bpm=135)
            cond_continuation(model, 'input/RM-P008.SMF_SYNC_BPM130.MID', fixed_velocity=[100, 80],
                              temperature=1.0, generation_length=384, n_samples=8, prompt_length=32, ins_ids=['drum', 'nondrum'], output_bpm=130)
    elif '_chord_mel' in model.save_name:

        is_chord_to_mel = '_rev' not in model.save_name
        if is_chord_to_mel:
            cond_continuation(model, 'input/ashover19_pitch_shift_2.mid', temperature=1.0, fixed_program=[0, 64],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-1', 'track-0'])
            cond_continuation(model, 'input/ashover28_pitch_shift_-2.mid', temperature=1.0, fixed_program=[0, 64],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-1', 'track-0'])
            cond_continuation(model, 'input/ashover37_pitch_shift_-2.mid', temperature=1.0, fixed_program=[0, 64],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-1', 'track-0'])
            cond_continuation(model, 'input/ashover46_pitch_shift_-2.mid', temperature=1.0, fixed_program=[0, 64],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-1', 'track-0'])
            cond_continuation(model, 'input/hpps13_pitch_shift_-2.mid', temperature=1.0, fixed_program=[0, 64],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-1', 'track-0'])
            cond_continuation(model, 'input/hpps22_pitch_shift_-2.mid', temperature=1.0, fixed_program=[0, 64],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-1', 'track-0'])
            cond_continuation(model, 'input/hpps31_pitch_shift_-2.mid', temperature=1.0, fixed_program=[0, 64],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-1', 'track-0'])
            cond_continuation(model, 'input/jigs108_pitch_shift_-2.mid', temperature=1.0, fixed_program=[0, 64],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-1', 'track-0'])
        else:
            cond_continuation(model, 'input/ashover19_pitch_shift_2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=24, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/ashover28_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/ashover37_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/ashover46_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=24, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/hpps13_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=24, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/hpps22_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=24, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/hpps31_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=24, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/jigs108_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/reelsm-q63_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/waltzes5_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-0', 'track-1'])
            cond_continuation(model, 'input/waltzes30_pitch_shift_-2.mid', temperature=1.0, fixed_program=[64, 0],
                              generation_length=384, n_samples=8, prompt_length=16, ins_ids=['track-0', 'track-1'])
    elif '_chords' in model.save_name:
        cond_continuation(model, 'input/la_beat/f947e58c78aa7c8055ef8dfc424ca22e_beat.mid', temperature=1.0,
                          generation_length=384, n_samples=8, prompt_length=16, ins_ids=['from-1', 'upto-0'])
        cond_continuation(model, 'input/la_beat/d9520bbf2bccd6424aa09f5694aa68f7_beat.mid', temperature=1.0,
                          generation_length=384, n_samples=8, prompt_length=16, ins_ids=['from-1', 'upto-0'])
        cond_continuation(model, 'input/la_beat/cf5f3bc804e474f4d0baf0c74656b042_beat.mid', temperature=1.0,
                          generation_length=384, n_samples=8, prompt_length=16, ins_ids=['from-1', 'upto-0'])
        cond_continuation(model, 'input/la_beat/394045b83f247bb862d7b09b1aacd78f_beat.mid', temperature=1.0,
                          generation_length=384, n_samples=8, prompt_length=16, ins_ids=['from-1', 'upto-0'])
        cond_continuation(model, 'input/la_beat/4261342f0970488e1381cb39867c48e1_beat.mid', temperature=1.0,
                          generation_length=384, n_samples=8, prompt_length=16, ins_ids=['from-1', 'upto-0'])
    else:
        raise ValueError('Unknown model type')


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('model_name', type=str)
    args = args.parse_args()
    model_name_list = args.model_name.split(',')
    for model_name in model_name_list:
        eval_model(model_name)