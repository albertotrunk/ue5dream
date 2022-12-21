import io, os, sys
import argparse
import unreal
from PIL import Image
import numpy as np
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import random

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt", type=str, nargs="?", default="this is unreal engine", help="the dream to render"
)

parser.add_argument(
    "--negative_prompt", type=str, nargs="?", default="", help="the nightmare to don't dream"
)

parser.add_argument(
    "--use_depth",  type=str, default="false"
)

parser.add_argument(
    "--strength",
    type=float,
    default=0.5,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)

parser.add_argument(
    "--engine",
    type=str,
    help="engine to use for inference",
    default="stable-diffusion-v1-5",
)

parser.add_argument(
    "--STABILITY_KEY",
    default="",
    help="beta.dreamstudio.ai->membership details->reveal",
)

opt = parser.parse_args()


# NB: host url is not prepended with \"https\" nor does it have a trailing slash.
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'

# To get your API key, visit https://beta.dreamstudio.ai/membership
os.environ['STABILITY_KEY'] = opt.STABILITY_KEY

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=False,
    engine=f"{opt.engine}" # Set the engine to use for generation.
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)

mask_img = None
init_img = Image.open(os.path.join(unreal.Paths.screen_shot_dir(), "dream_color.png"))

if opt.use_depth == "true":
    mask_img = Image.open(os.path.join(unreal.Paths.screen_shot_dir(), "dream_mask.png"))

seed = [random.randrange(0, 4294967295)]

# the object returned is a python generator
answers = stability_api.generate(
    prompt= [generation.Prompt(text=opt.prompt,parameters=generation.PromptParameters(weight=1)),
    generation.Prompt(text=opt.negative_prompt,parameters=generation.PromptParameters(weight=-1.3))], # Negative prompting is now possible via the API, simply assign a negative weight to a prompt.
    # In the example above we are combining a mountain landscape with the style of thomas kinkade, and we are negative prompting trees out of the resulting concept.
    # When determining prompt weights, the total possible range is [-10, 10] but we recommend staying within the range of [-2, 2].
    seed=seed, # if provided, specifying a random seed makes results deterministic
    steps=20, # defaults to 30 if not specified
    height=768,
    width=768,
    start_schedule=opt.strength, # this controls the "strength" of the prompt relative to the init image
    cfg_scale=9,
    init_image=init_img,
    mask_image=mask_img,
    sampler=generation.SAMPLER_DDIM,
    # Choose which sampler we want to denoise our generation with.
    # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
    samples=2
    #"guidance_preset"=generation.GUIDANCE_PRESET_SLOW,
    #"guidance_strength"=2.5,
    #"guidance_prompt"="unreal engine 5",
)

with unreal.ScopedSlowTask(3, "Stability AI is dreaming!") as slow_task:
    slow_task.make_dialog(True)
    # iterating over the generator produces the api response
    slow_task.enter_progress_frame(1)
    for resp in answers:
        if slow_task.should_cancel():
            unreal.log_warning("Task cancelled")
            break
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                unreal.log_warning(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
                exit()
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                path = os.path.join(unreal.Paths.screen_shot_dir(), f"dream-{opt.strength}-{artifact.seed}.png")
                img.save(path)
                unreal.log(f"Saved image to {path}")
                img.show()

    slow_task.enter_progress_frame(2)

slow_task.enter_progress_frame(3)
