# to run this, you need a python 3.12.x environment or later as either a conda
# environment or a python venv and pip install the requirements.txt file
# (pip install -r requirements.txt)
import gc

import torch
import os
import random
import PIL.PngImagePlugin
import diffusers.image_processor
import json
import time
from compel import Compel, ReturnedEmbeddingsType

# set your hugging face token here
os.environ["HF_TOKEN"] = ""

# the model you want this script to use.  must match the model key in
# available_models.json
RENDER_MODEL = "flux-1-dev"

# if you have a GPU, set GPU_DEVICE to the GPU you want to use.  Usually
# cuda:0 is fine, but if you have more than one gpu, then set the GPU number to
# use, i.e. cuda:1, cuda:2, etc.  The CPU device is never anything but cpu.
GPU_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"

# if this is set to false, then the seed used for each image is based on the
# current time epoch, if set to true, then a random seed in 32 bit space is
# randomly selected.
USE_RANDOM_SEED = False

# the number of images we want to generate per prompt variation
NUMBER_IMAGES_TO_GENERATE = 1

# Below are the prompt fragments.  The script will run through and permutate every
# combination of fragments you have set.  If you don't want to use a particular
# fragment, just set it with a single empty string and the script will ignore it.
# each fragment can have as many list items as you want to permutate through,
# though, be careful, as you can very quickly balloon out to millions of images.

# the main subject to be generated, include a general description
SUBJECTS = [
    ""
]

# if the subject has any clothing, specify variations here
CLOTHING = [
    ""
]

# clothing or image color palettes, modify to suit your needs
COLOR_PALETTES = [
    ""
    "reds",
    "oranges",
    "yellows",
    "greens",
    "cyans",
    "blues",
    "navy blues",
    "purples",
    "pinks",
    "magentas",
    "whites",
    "blacks",
    "tans",
    "golds",
    "silvers"
]

# the style of the image, i.e. realistic photograph, oil painting, etc.
STYLES = [
    "realistic, photorealistic, photograph",
]

# the image aesthetics
AESTHETICS = [
    ""
]

# the composition of the image.  I.e. where is the subject, describe the
# surroundings
COMPOSITIONS = [
    "",
]

# lighting variations, self explanitory, modify to meet your needs.
LIGHTINGS = [
    "bright mid-day overhead sun",
    "late-day sun with long shadows",
    "golden hour",
    "blue hour",
    "nighttime",
    "overcast diffuse light"
]

# the image mood or atmosphere, subjective
MOOD_ATMOSPHERES = [
    "",
]

# technical image details, modify to meet your needs
TECHNICAL_DETAILS = [
    "sharp details, 4k, 8k, high resolution, high quality",
]

# camera distances, modify to meet your needs
CAMERA_DISTANCES = [
    "extreme close up shot",
    "close up shot",
    "medium close up shot, showing just the head and shoulders",
    "medium shot, shot from the waist up",
    "medium long shot, medium wide shot, medium full shot, shot from the knees up",
    "full body shot, long shot, wide shot, shot from the feet up",
    "extreme long shot, extreme wide shot"
]

# if the subject has poses, modify to meet your needs
POSES = [
    "front view",
    "side view",
    "back view, from behind",
]

# any additional elements that you want to have variations of, modify to meet
# needs
ADDITIONAL_ELEMENTS = [""]

# we only have one negative prompt, not all the models even use a negative
# prompt
NEGATIVE_PROMPT = "nsfw, nude, low detail, low quality, low resolution, blurry"


def process_subject(configuration):
    for subject in SUBJECTS:
        configuration["subject"] = subject
        process_clothes(configuration)


def process_clothes(configuration):
    for clothes in CLOTHING:
        configuration["clothes"] = clothes
        process_style(configuration)


def process_style(configuration):
    for style in STYLES:
        configuration["style"] = style
        process_aesthetic(configuration)


def process_aesthetic(configuration):
    for aesthetic in AESTHETICS:
        configuration["aesthetic"] = aesthetic
        process_composition(configuration)


def process_composition(configuration):
    for composition in COMPOSITIONS:
        configuration["composition"] = composition
        process_lighting(configuration)


def process_lighting(configuration):
    for lighting in LIGHTINGS:
        configuration["lighting"] = lighting
        process_color_palette(configuration)


def process_color_palette(configuration):
    for color_palette in COLOR_PALETTES:
        configuration["color_palette"] = color_palette
        process_mood_atmosphere(configuration)


def process_mood_atmosphere(configuration):
    for mood_atmosphere in MOOD_ATMOSPHERES:
        configuration["mood_atmosphere"] = mood_atmosphere
        process_technical_details(configuration)


def process_technical_details(configuration):
    for technical_detail in TECHNICAL_DETAILS:
        configuration["technical_detail"] = technical_detail
        process_camera_distance(configuration)


def process_camera_distance(configuration):
    for camera_distance in CAMERA_DISTANCES:
        configuration["camera_distance"] = camera_distance
        process_pose(configuration)


def process_pose(configuration):
    for pose in POSES:
        configuration["pose"] = pose
        process_additional_elements(configuration)


def process_additional_elements(configuration):
    for additional_element in ADDITIONAL_ELEMENTS:
        configuration['additional_element'] = additional_element
        generate_images(configuration)


def generate_images(configuration):
    model = get_model()
    # set up our image generation parameters
    seed_generator = torch.Generator(device=CPU_DEVICE)

    i = 0
    while i < NUMBER_IMAGES_TO_GENERATE:
        if USE_RANDOM_SEED:
            seed = random.randint(1, 4294967295)
        else:
            seed = int(time.time())

        # generate the primary image
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        seed_generator.manual_seed(seed)
        seed_generator.manual_seed(seed)

        file_base_name = f"{RENDER_MODEL}_{seed}"
        file_name = f"{file_base_name}.png"

        print()
        print(f"Generating {file_name} ({i + 1}-{NUMBER_IMAGES_TO_GENERATE}) "
              f"with model {model['name']}")
        print()

        # set file some metadata
        image_info = PIL.PngImagePlugin.PngInfo()
        image_info.add_text("model", model['name'])
        image_info.add_text("seed", f"{seed}")
        image_info.add_text("num_inference_steps", f"{model['num_inference_steps']}")
        if "guidance_scale" in model:
            image_info.add_text("guidance_scale", f"{model['guidance_scale']}")
        if "true_cfg_scale" in model:
            image_info.add_text("true_cfg_scale", f"{model['true_cfg_scale']}")
        image_info.add_text("prompt_type", "string")
        image_info.add_text("negative_prompt", NEGATIVE_PROMPT)
        image_info.add_text("subject", configuration['subject'])
        image_info.add_text("clothing", configuration['clothes'])
        image_info.add_text("style", configuration['style'])
        image_info.add_text("aesthetic", configuration['aesthetic'])
        image_info.add_text("composition", configuration['composition'])
        image_info.add_text("lighting", configuration['lighting'])
        image_info.add_text("color_palette", configuration['color_palette'])
        image_info.add_text("mood_atmosphere", configuration['mood_atmosphere'])
        image_info.add_text("technical_detail", configuration['technical_detail'])
        image_info.add_text("camera_distance", configuration['camera_distance'])
        image_info.add_text("pose", configuration['pose'])
        image_info.add_text("additional_element", configuration['additional_element'])

        if RENDER_MODEL == 'flux-1-dev' or RENDER_MODEL == 'flux-1-schnell' or RENDER_MODEL == 'flux-1-krea' or RENDER_MODEL == 'sd-3-5-large' or RENDER_MODEL == 'sd-3-5-medium' or RENDER_MODEL == 'sd-3-0':
            positive_prompt = ""
            if len(configuration['technical_detail']) > 0:
                positive_prompt += f"technical details: {configuration['technical_detail']}"
            if len(configuration['subject']) > 0:
                positive_prompt += f", subject: {configuration['subject']}"
            if len(configuration['clothes']) > 0:
                positive_prompt += f", clothing: {configuration['clothes']}"
            if len(configuration['color_palette']) > 0:
                positive_prompt += f", with a color palette of: {configuration['color_palette']}"
            if len(configuration['composition']) > 0:
                positive_prompt += f", composition: {configuration['composition']}"
            if len(configuration['camera_distance']) > 0:
                positive_prompt += f", camera distance: {configuration['camera_distance']}"
            if len(configuration['pose']) > 0:
                positive_prompt += f", pose: {configuration['pose']}"
            if len(configuration['style']) > 0:
                positive_prompt += f", style: {configuration['style']}"
            if len(configuration['aesthetic']) > 0:
                positive_prompt += f", aesthetic: {configuration['aesthetic']}"
            if len(configuration['lighting']) > 0:
                positive_prompt += f", lighting: {configuration['lighting']}"
            if len(configuration['mood_atmosphere']) > 0:
                positive_prompt += f", mood and atmosphere: {configuration['mood_atmosphere']}"
            if len(configuration['additional_element']) > 0:
                positive_prompt += f", additional elements: {configuration['additional_element']}"

            image_info.add_text("positive_prompt", positive_prompt)
            print(f"positive prompt: {positive_prompt}")
            print()

            if RENDER_MODEL == 'sd-3-0':
                primary_image = configuration['primary_pipe'](
                    prompt=positive_prompt,
                    prompt_3=positive_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    max_sequence_length=model['max_sequence_length'],
                    num_inference_steps=model['num_inference_steps'],
                    guidance_scale=model['guidance_scale'],
                    width=model['native_width'],
                    height=model['native_height'],
                    generator=seed_generator
                ).images[0]
            elif RENDER_MODEL == 'sd-3-5-medium' or RENDER_MODEL == 'sd-3-5-large':
                primary_image = configuration['primary_pipe'](
                    prompt=positive_prompt,
                    prompt_3=positive_prompt,
                    max_sequence_length=model['max_sequence_length'],
                    num_inference_steps=model['num_inference_steps'],
                    guidance_scale=model['guidance_scale'],
                    width=model['native_width'],
                    height=model['native_height'],
                    generator=seed_generator
                ).images[0]
            else:
                primary_image = configuration['primary_pipe'](
                    prompt=positive_prompt,
                    max_sequence_length=model['max_sequence_length'],
                    num_inference_steps=model['num_inference_steps'],
                    guidance_scale=model['guidance_scale'],
                    width=model['native_width'],
                    height=model['native_height'],
                    generator=seed_generator
                ).images[0]

        elif RENDER_MODEL == 'qwen-image':
            positive_prompt = ""
            if len(configuration['technical_detail']) > 0:
                positive_prompt += f"technical details: {configuration['technical_detail']}"
            if len(configuration['subject']) > 0:
                positive_prompt += f", subject: {configuration['subject']}"
            if len(configuration['clothes']) > 0:
                positive_prompt += f", clothing: {configuration['clothes']}"
            if len(configuration['color_palette']) > 0:
                positive_prompt += f", with a color palette of: {configuration['color_palette']}"
            if len(configuration['composition']) > 0:
                positive_prompt += f", composition: {configuration['composition']}"
            if len(configuration['camera_distance']) > 0:
                positive_prompt += f", camera distance: {configuration['camera_distance']}"
            if len(configuration['pose']) > 0:
                positive_prompt += f", pose: {configuration['pose']}"
            if len(configuration['style']) > 0:
                positive_prompt += f", style: {configuration['style']}"
            if len(configuration['aesthetic']) > 0:
                positive_prompt += f", aesthetic: {configuration['aesthetic']}"
            if len(configuration['lighting']) > 0:
                positive_prompt += f", lighting: {configuration['lighting']}"
            if len(configuration['mood_atmosphere']) > 0:
                positive_prompt += f", mood and atmosphere: {configuration['mood_atmosphere']}"
            if len(configuration['additional_element']) > 0:
                positive_prompt += f", additional elements: {configuration['additional_element']}"

            image_info.add_text("positive_prompt", positive_prompt)
            print(f"positive prompt: {positive_prompt}")
            print()

            primary_image = configuration['primary_pipe'](
                prompt=positive_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=model['num_inference_steps'],
                true_cfg_scale=model['true_cfg_scale'],
                width=model['native_width'],
                height=model['native_height'],
                generator=seed_generator
            ).images[0]

        else:
            positive_prompt = ""
            if len(configuration['technical_detail']) > 0:
                positive_prompt += f"{configuration['technical_detail']}"
            if len(configuration['subject']) > 0:
                positive_prompt += f", {configuration['subject']}"
            if len(configuration['clothes']) > 0:
                positive_prompt += f", {configuration['clothes']}"
            if len(configuration['color_palette']) > 0:
                positive_prompt += f", with a color palette of {configuration['color_palette']}"
            if len(configuration['composition']) > 0:
                positive_prompt += f", {configuration['composition']}"
            if len(configuration['camera_distance']) > 0:
                positive_prompt += f", {configuration['camera_distance']}"
            if len(configuration['pose']) > 0:
                positive_prompt += f", {configuration['pose']}"
            if len(configuration['style']) > 0:
                positive_prompt += f", {configuration['style']}"
            if len(configuration['aesthetic']) > 0:
                positive_prompt += f", {configuration['aesthetic']}"
            if len(configuration['lighting']) > 0:
                positive_prompt += f", {configuration['lighting']}"
            if len(configuration['mood_atmosphere']) > 0:
                positive_prompt += f", {configuration['mood_atmosphere']}"
            if len(configuration['additional_element']) > 0:
                positive_prompt += f", {configuration['additional_element']}"

            image_info.add_text("positive_prompt", positive_prompt)
            print(f"positive prompt: {positive_prompt}")
            print()

            if RENDER_MODEL == 'sdxl-1-0' or RENDER_MODEL == 'rubb-xl-1-0':
                positive_compel = Compel(
                    tokenizer=[configuration['primary_pipe'].tokenizer,
                               configuration['primary_pipe'].tokenizer_2],
                    text_encoder=[configuration['primary_pipe'].text_encoder,
                                  configuration['primary_pipe'].text_encoder_2],
                    requires_pooled=[False, True],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    truncate_long_prompts=False)

                negative_compel = Compel(
                    tokenizer=[configuration['primary_pipe'].tokenizer,
                               configuration['primary_pipe'].tokenizer_2],
                    text_encoder=[configuration['primary_pipe'].text_encoder,
                                  configuration['primary_pipe'].text_encoder_2],
                    requires_pooled=[False, True],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    truncate_long_prompts=False)

                [positive_embed, positive_pooled] = positive_compel([positive_prompt])
                [negative_embed, negative_pooled] = negative_compel([NEGATIVE_PROMPT])

                [positive_embed,
                 negative_embed] = positive_compel.pad_conditioning_tensors_to_same_length(
                    [positive_embed, negative_embed])

                primary_image = configuration['primary_pipe'](
                    prompt_embeds=positive_embed,
                    pooled_prompt_embeds=positive_pooled,
                    negative_prompt_embeds=negative_embed,
                    negative_pooled_prompt_embeds=negative_pooled,
                    num_inference_steps=model['num_inference_steps'],
                    guidance_scale=model['guidance_scale'],
                    width=model['native_width'],
                    height=model['native_height'],
                    generator=seed_generator
                ).images[0]

                positive_embed = None
                negative_embed = None
                positive_pooled = None
                negative_pooled = None
                gc.collect()

            else:
                positive_compel = Compel(
                    tokenizer=configuration['primary_pipe'].tokenizer,
                    text_encoder=configuration['primary_pipe'].text_encoder,
                    truncate_long_prompts=False)

                negative_compel = Compel(
                    tokenizer=configuration['primary_pipe'].tokenizer,
                    text_encoder=configuration['primary_pipe'].text_encoder,
                    truncate_long_prompts=False)

                positive_embed = positive_compel([positive_prompt])
                negative_embed = negative_compel([NEGATIVE_PROMPT])

                [positive_embed, negative_embed] = positive_compel.pad_conditioning_tensors_to_same_length([positive_embed, negative_embed])

                primary_image = configuration['primary_pipe'](
                    prompt_embeds=positive_embed,
                    negative_prompt_embeds=negative_embed,
                    num_inference_steps=model['num_inference_steps'],
                    guidance_scale=model['guidance_scale'],
                    width=model['native_width'],
                    height=model['native_height'],
                    generator=seed_generator
                ).images[0]

                positive_embed = None
                negative_embed = None
                gc.collect()

        if primary_image is not None:
            print()
            print(f"Saving {file_name}")
            primary_image.save(os.path.join(configuration['images_dir'], file_name),
                               pnginfo=image_info)

        i += 1


def get_model():
    return json.loads(open(os.path.join(os.getcwd(), "available_models.json")).read())[RENDER_MODEL]


configuration = {
    'device': GPU_DEVICE if torch.cuda.is_available() else CPU_DEVICE,
    'models_dir': os.path.join(os.getcwd(), "models"),
    'images_dir': os.path.join(os.getcwd(), "generated_images")
}

print(f"generating images using {configuration['device']}")

# set our float type based on the device type.
configuration['float_type'] = torch.float16 if configuration['device'] == GPU_DEVICE else torch.float32

# set up our directories
os.makedirs(configuration['images_dir'], exist_ok=True)

model = get_model()
print(f"{model['name']} has been selected")
if RENDER_MODEL == 'qwen-image':
    configuration['primary_pipe'] = diffusers.DiffusionPipeline.from_pretrained(
        model['name'],
        torch_dtype=configuration['float_type'],
        cache_dir=configuration['models_dir'])
else:
    configuration['primary_pipe'] = diffusers.AutoPipelineForText2Image.from_pretrained(
        model['name'],
        torch_dtype=configuration['float_type'],
        cache_dir=configuration['models_dir'])

if configuration['device'] == GPU_DEVICE and model["cpu_offload"] is True:
    print("enabling CPU offloading")
    configuration['primary_pipe'].enable_sequential_cpu_offload()
else:
    print("disabling CPU offloading")
    configuration['primary_pipe'].to(configuration['device'])

# kick off stepping through each variation starting with process_subject()
process_subject(configuration)
