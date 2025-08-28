# diffusers_generate_images
A Stable Diffusion image generation script that generates images based on prompt permutations
----
Mostly a research tool, but can be useful to explore a models prompt adherence, or how well it handles different prompt variations.  Can also be used to generate images for training datasets, i.e. images from later models used to fine tune earlier models.

## Usage
1.  You need either a conda environment or python virtual environment of at least python 3.12
2.  Activate your environment
3.  pip install -r requirements.txt
4.  Modify `diffusers_generate_images.py` to add your hugging face token and set up your prompt variations you want to permute through.
5.  `python diffusers_generate_images.py` - it will run until finished, there will be a `models` subdirectory where the model files are stored, and a `generated_images` subdirectory where your generated images are store.

----
You may want to run the script in a `tmux` session on a dedicated system as it can a long time to run depending on how many permutations you have set up.
