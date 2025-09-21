# Uncurated Demo README

This page contains the uncurated demos for the paper [Versatile Symbolic Music-for-Music Modeling via Function Alignment](https://arxiv.org/abs/2506.15548).

Root folder: ``prompt{length}_cp_transformer_{model_name}_batch_size_8_{dataset_name}_{task_name}``, where
* ``length``: the length of the input prompt (in number of 16th notes)
* ``model_name``: the name of the adapter
  * ``prompt_lora``: self-attentive adapter
  * ``yinyang``: cross-attentive adapter
  * ``cocomulla``: Coco-Mulla adapter (baseline)
  * ``seq2seq``: Encoder-decoder modeling (trained from scratch).
* ``dataset_name``: the name of the dataset used in fine-tuning
  * ``la_cp16_v2``: Los Angeles MIDI dataset (subset with drum tracks)
  * ``nottingham_cp8_v2``: Nottingham dataset
* ``task_name``: the name of the task
  * ``mel_to_chord``: melody to chord generation
  * ``chord_to_mel``: chord to melody generation
  * ``drum_to_others``: drum to others generation
  * ``others_to_drum``: others to drum generation
  * ``midi_analysis``: MIDI analysis