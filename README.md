# Versatile Symbolic Music-for-Music Modeling via Function Alignment

## THIS PAGE IS STILL UNDER CONSTRUCTION. PLEASE CHECK BACK LATER.

Welcome to the official repo for the ISMIR 2025 Paper [Versatile Symbolic Music-for-Music Modeling via Function Alignment](https://arxiv.org/abs/2506.15548)!

<img width="2400" height="1087" alt="image" src="https://github.com/user-attachments/assets/217b4679-3d73-4470-b12a-27cb41fe4925" />

## What is Music-for-Music Modeling?

Music for music refers to represent both input/output sequences (including labels like chords, beats, textures, keys, structures) using the music modality itself. This unifies music understanding (music -> labels) and conditional generation (labels -> music) into the same form (music -> music).

## What is function alignment?

Function alignment is a [new theory of mind](https://arxiv.org/abs/2503.21106) that attributes human-like intelligence towards dynamic synergy among interacting agent (i.e., Language Models).

<img width="1198" height="645" alt="image" src="https://github.com/user-attachments/assets/817fb6c7-ecbd-42bb-987c-ae0b3e16cbe7" />


## What did we do?

We use function alignment to model music for music tasks in a unified manner. This includes:

### Conditional Generation Tasks
* Melody to chord
* Chord to melody
* Drums to others
* Others to drum

### Music Analysis Tasks
* MIDI chord & metrical structure analysis

All these tasks share (1) the same foundation model and (2) the same adapter architecture, but are fine-tuned on different datasets.

# Pretrained Models

All pretrained models are available at [Google Drive](https://drive.google.com/drive/folders/1E_gzGgc4Pzd-jpMxOaLEewuvZ35l3jeU?usp=sharing).

It contains the following files:

* ``cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt``: The foundation model (Roformer) trained on 16th-note quantized MIDI sequences.
* ``mel_to_chord``: Models for melody to chord generation.
* ``chord_to_mel``: Models for chord to melody generation.
* ``drum_to_others``: Models for drum to others generation.
* ``others_to_drum``: Models for others to drum generation.
* ``midi_analysis``: Models for MIDI analysis.

Download the models and put them in the ``ckpt/`` folder. Keep the subfolder structure.

You must download the foundation model to use other models.

# Downstream Models

To use a downstream model, call the ``cp_transformer_yinyang_inference.py`` script. For example:

```bash
python cp_transformer_yinyang_inference.py ckpt/mel_to_chord/cp_transformer_yinyang_v5.1_lora_batch_8_nottingham_cp8_v2_chord_mel_rev_mask0.0-10-step1.epoch=last.ckpt
```

The examples (input MIDI files and config) for each model is hardcoded in the ``cp_transformer_yinyang_inference.py`` file.

# Foundation Model

The foundation model is a Roformer model trained on 16th-note quantized MIDI sequences.

To use the pretrained model for inference (e.g., continuation), you can directly run the following command:

```bash
python cp_transformer_inference.py ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt
```

You may use your own MIDI files, just ensure the beats (deduced from tempo changes) are correct in the MIDI file.

# Future Works

## Unsupervised MIDI Chord and Key Estimation

One of the future works, unsupervised MIDI chord and key estimation is also open-sourced on [Github](https://github.com/instr3/MIDI-AutoLabel-Dataset).

The work uses the same foundation model as this repo, but is fine-tuned in a totally unsupervised way (with pseudo-labels).

Though this work does not use function alignment, it is still a music-for-music modeling task and can be implemented via function alignment in the future.

## Other Future Works

We are currently working on function alignment for more tasks, as well as a better foundation model for symbolic music.

If you are interested in helping us, please feel free to contact us at jj2731@nyu.edu.

