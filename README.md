# Install

The project relies on `uv` to manage its dependencies.

To create virtual env an activate it:
1) `uv sync`
2) `source .venv/bin/activate`

# Project structure

Directories 1-5 contain the scripts by the IFC paper's authors. Our (Appsilon's) work is entirely restricted to directory `6 - Appsilon`. The directory structure inside mimics the original 1-5 directories, since we are basing our approach on what the authors' did, at least to a large degree.

When performing experiments it is recommended to create logical subfolders that mark the different chapters in our work. For instance:

.
├── Fine-tuning ConvNext
├── Fine-tuning ConvNext, new Augments
├── Fine-tuning ViT LoRA
└── Reproducing ResNet18 results


The following hierarchy represents:
1) Trying to reproduce authors' original ResNet results
2) Switching to ConvNext to try and improve them
3) Another switch to ViT with LoRA applied
4) Change of direction - trying augments of our own design, once again with ConvNext