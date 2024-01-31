from diffusers import DiffusionPipeline
from concurrent.futures import ThreadPoolExecutor
import argparse
import torch
import datasets
import os


ALL_CKPTS = [
    "runwayml/stable-diffusion-v1-5",
    "segmind/SSD-1B",
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
]


def load_dataset():
    dataset = datasets.load_dataset("nateraw/parti-prompts", split="train")
    return dataset


def load_pipeline(args):
    pipeline = DiffusionPipeline.from_pretrained(args.pipeline_id, torch_dtype=torch.float16).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def generate_images(args, dataset, pipeline):
    all_images = []
    for i in range(0, len(dataset), args.chunk_size):
        images = pipeline(
            dataset[i : i + args.chunk_size]["Prompt"],
            num_inference_steps=args.num_inference_steps,
            generator=torch.manual_seed(2024),
        ).images
        all_images.extend(images)
    return all_images


def serialize_image(image, path):
    image.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_id", default="runwayml/stable-diffusion-v1-5", type=str, choices=ALL_CKPTS)
    parser.add_argument("--num_inference_steps", default=30, type=int)
    parser.add_argument("--chunk_size", default=2, type=int)
    parser.add_argument("--root_img_path", default="sdv15", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dataset = load_dataset()
    pipeline = load_pipeline(args)
    images = generate_images(args, dataset, pipeline)
    image_paths = [os.path.join(args.root_img_path, f"{i}.jpg") for i in range(len(images))]

    if not os.path.exists(args.root_img_path):
        os.makedirs(args.root_img_path)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        executor.map(serialize_image, images, image_paths)
