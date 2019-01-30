# Machine Learning Code and info I tend to reuse

* nn
  - Highway Network
  - Reference Encoder (Towards End-to-End Prosody Transfer)
* gym 
  - test OpenAI gym script
* datasets
  - Kaggle Speech Word Classification
  - datasets (pytorch)
    - mnist
* utils
  - Spectrogram
  - conv\_out\_shape

## Install fixes

#### opencv3
* Installation conda:
  - On Ubuntu 18.04 (kde neon) in chronological order
  - `conda install -c menpo opencv3 `
  - `ImportError: libpng12.so.0: cannot open shared object file: No such file or directory`
  - `sudo apt install libpng12-dev`
  - `ImportError: /usr/lib/x86_64-linux-gnu/libpangoft2-1.0.so.0: undefined symbol: hb_font_funcs_set_variation_glyph_func`
  - `conda install -c conda-forge pango`  -> works



## Inspiration
* [Schmidhuber](http://people.idsia.ch/~juergen/)
  - [World model](https://worldmodels.github.io/)
  - All his NeurIPS talks
  - RNN's are turing complete
* [Colah Blog](http://colah.github.io/)
* [Agustinus Kristiadi](https://wiseodd.github.io/about/)

