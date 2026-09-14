# coding=utf-8
# Copyright 2024 The HuggingFace, NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is heavily inspired by https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py

import itertools
import json
import math
import os
import random
import re
from functools import partial
from typing import Any, Dict, List, Optional, Union

from PIL import Image

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

import torch
import webdataset as wds
import yaml
from braceexpand import braceexpand
from torch.utils.data import default_collate
from torchvision import transforms
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

person_token = ["a person", "someone", "somebody"]


def replace_person_token(t):
    "Used for CC12M"
    t = re.sub(r"<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    while "<person>" in t:
        t = t.replace("<person>", f" {random.choices(person_token)} ", 1)
    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def image_transform(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    sample["images"] = image
    return sample


def _image_to_tensor(image, resolution=256):
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image)!r}.")
    image = image.convert("RGB")
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


def _mask_to_tensor(mask, resolution=256):
    if mask is None:
        return torch.empty(0)
    if not isinstance(mask, Image.Image):
        raise TypeError(f"Expected PIL mask, got {type(mask)!r}.")
    mask = mask.convert("L")
    mask = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST)(mask)
    mask = transforms.CenterCrop((resolution, resolution))(mask)
    mask = transforms.ToTensor()(mask)
    return (mask > 0.5).float()


def _decode_edit_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return _decode_edit_text(json.loads(stripped))
            except json.JSONDecodeError:
                return stripped
        return stripped
    if isinstance(value, dict):
        for key in (
            "edit_instruction", "instruction", "edit", "prompt", "text",
            "caption", "input", "command",
        ):
            if key in value:
                return _decode_edit_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return " ".join(_decode_edit_text(item) for item in value if _decode_edit_text(item))
    return str(value)


def _find_by_suffix(sample: Dict[str, Any], suffixes: List[str]):
    for key, value in sample.items():
        if key.startswith("__"):
            continue
        key_l = key.lower()
        if key_l in suffixes or any(key_l.endswith(suffix) for suffix in suffixes):
            return value
    return None


def remove_prefix(caption):
    caption = caption.replace('The image features ', '').replace('The image presents ', '').replace(
        "The image you've sent is, ", '').replace("In the center of the image, ", '').replace(
        "The image showcases ", '').replace("The image is ", '').replace(
        "The image captures ", '').replace("In the given image ", '').replace(
        "The image portrays ", '').replace("In the image, ", '').replace("In this image, we see ", '').replace(
        "The image depicts ", '').replace("This is ", '').replace("In this image, ", '').replace(
        "This image captures ", '')

    return caption


class Text2ImageDataset:
    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str]],
            tokenizer: PreTrainedTokenizer,
            max_seq_length: int,
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            resolution: int = 256,
            shuffle_buffer_size: int = 1000,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            external_caption_path: Optional[str] = '',
            external_journeydb_caption_path: Optional[str] = '',
            external_laion12m_caption_path: Optional[str] = '',
            external_cc12m_caption_path: Optional[str] = '',
            is_captioning: bool = False,
            add_caption_prompt: bool = False,
            long_caption: bool = True,
    ):
        if f"{train_shards_path_or_url}.yaml" in os.listdir('./configs'):
            with open(f"./configs/{train_shards_path_or_url}.yaml") as f:
                train_shards_path_or_url = yaml.safe_load(f)
        self.long_caption = long_caption
        self.external_caption_path = external_caption_path
        self.external_journeydb_caption_path = external_journeydb_caption_path
        self.external_laion12m_caption_path = external_laion12m_caption_path
        self.external_cc12m_caption_path = external_cc12m_caption_path
        self.is_captioning = is_captioning
        self.add_caption_prompt = add_caption_prompt
        if self.add_caption_prompt:
            with open("./training/questions.json") as f:
                self.caption_prompt = json.load(f)
                self.caption_prompt = ['USER: \n' + prompt + ' ASSISTANT:' for prompt in self.caption_prompt]
        else:
            self.caption_prompt = None

        if external_journeydb_caption_path != '':
            with open(external_journeydb_caption_path) as file:
                self.journeydb_caption = json.load(file)
        else:
            self.journeydb_caption = None

        def tokenize(text):
            if tokenizer is not None:
                text = replace_person_token(text)
                input_ids = tokenizer(
                    text, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                return input_ids[0]
            else:
                return text

        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        if external_caption_path != '':
            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.map(self.load_external_caption, handler=wds.ignore_and_continue),
                wds.rename(
                    images="jpg;png;jpeg;webp",
                    input_ids="text;txt;caption",
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys(set(["images", "input_ids"]))),
                wds.map(partial(image_transform, resolution=resolution), handler=wds.warn_and_continue),
                wds.map_dict(
                    input_ids=tokenize,
                    handler=wds.warn_and_continue,
                ),
            ]
        else:
            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.rename(
                    images="jpg;png;jpeg;webp",
                    input_ids="text;txt;caption",
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys(set(["images", "input_ids"]))),
                wds.map(partial(image_transform, resolution=resolution), handler=wds.warn_and_continue),
                wds.map_dict(
                    input_ids=tokenize,
                    handler=wds.warn_and_continue,
                ),
            ]

        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            # num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            # persistent_workers=0,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    def load_external_caption(self, sample):

        if 'txt' not in sample.keys():
            sample['txt'] = ''

        if 'SA1B' in sample['__key__']:
            captionf = f"{self.external_caption_path}/{sample['__key__'].split('/')[-1]}.txt"
            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.readlines()[0].replace('\n', '')
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + ' ' + captions
                else:
                    sample['txt'] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample['txt'] = captions.split('.')[0]
                else:
                    sample['txt'] = captions

                sample['txt'] = remove_prefix(sample['txt'])

            return sample

        elif 'laion' in sample['__url__']:
            captionf = f"{self.external_laion12m_caption_path}/{sample['__url__'].split('/')[-1].split('.')[0]}/{sample['__key__']}.caption"
            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.readlines()[0].replace('\n', '')
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + ' ' + captions
                else:
                    sample['txt'] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample['txt'] = captions.split('.')[0]
                else:
                    sample['txt'] = captions

                sample['txt'] = remove_prefix(sample['txt'])

            return sample

        elif 'cc12m' in sample['__url__']:
            captionf = f"{self.external_cc12m_caption_path}/{sample['__url__'].split('/')[-1].split('.')[0]}/{sample['__key__'].split('/')[-1]}.caption"
            if os.path.exists(captionf):
                with open(captionf, "r") as reader:
                    captions = reader.readlines()[0].replace('\n', '')
            else:
                captions = ""

            # for captioning
            if self.is_captioning:
                if self.add_caption_prompt is not None:
                    prompt = random.sample(self.caption_prompt, 1)[0]
                    sample['txt'] = prompt + ' ' + captions
                else:
                    sample['txt'] = captions
            # for generation
            else:
                # randomly choose short and long captions
                if random.random() < 0.5:
                    sample['txt'] = captions.split('.')[0]
                else:
                    sample['txt'] = captions
                sample['txt'] = remove_prefix(sample['txt'])

            return sample

        elif self.journeydb_caption is not None and sample['__key__'] in self.journeydb_caption:
            sample['txt'] = random.sample(self.journeydb_caption[sample['__key__']], 1)[0]
            return sample

        else:
            return sample

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


class ImageEditingDataset:
    """WebDataset loader for source-image/edit-instruction/target-image triplets.

    Supported tar suffixes include:
      source image: src/source/input/original/before
      target image: target/edited/output/tgt/after
      instruction: instruction/edit/prompt/text/txt/json
      optional edit mask: mask/edit_mask/local_mask
    """

    SOURCE_SUFFIXES = [
        "src.jpg", "src.jpeg", "src.png", "src.webp",
        "source.jpg", "source.jpeg", "source.png", "source.webp",
        "input.jpg", "input.jpeg", "input.png", "input.webp",
        "original.jpg", "original.jpeg", "original.png", "original.webp",
        "before.jpg", "before.jpeg", "before.png", "before.webp",
    ]
    TARGET_SUFFIXES = [
        "target.jpg", "target.jpeg", "target.png", "target.webp",
        "tgt.jpg", "tgt.jpeg", "tgt.png", "tgt.webp",
        "edited.jpg", "edited.jpeg", "edited.png", "edited.webp",
        "output.jpg", "output.jpeg", "output.png", "output.webp",
        "after.jpg", "after.jpeg", "after.png", "after.webp",
    ]
    INSTRUCTION_SUFFIXES = [
        "instruction.txt", "edit.txt", "prompt.txt", "text.txt",
        "instruction", "edit", "prompt", "text", "txt", "json",
    ]
    MASK_SUFFIXES = [
        "mask.jpg", "mask.jpeg", "mask.png", "mask.webp",
        "edit_mask.jpg", "edit_mask.jpeg", "edit_mask.png", "edit_mask.webp",
        "local_mask.jpg", "local_mask.jpeg", "local_mask.png", "local_mask.webp",
    ]

    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str]],
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            resolution: int = 256,
            shuffle_buffer_size: int = 1000,
            pin_memory: bool = False,
            persistent_workers: bool = False,
    ):
        if f"{train_shards_path_or_url}.yaml" in os.listdir('./configs'):
            with open(f"./configs/{train_shards_path_or_url}.yaml") as f:
                train_shards_path_or_url = yaml.safe_load(f)

        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.map(self.to_edit_triplet, handler=wds.warn_and_continue),
            wds.map(filter_keys(set(["source_images", "target_images", "edit_instructions", "edit_masks"]))),
        ]

        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        self.resolution = resolution

    def to_edit_triplet(self, sample):
        source = _find_by_suffix(sample, self.SOURCE_SUFFIXES)
        target = _find_by_suffix(sample, self.TARGET_SUFFIXES)
        instruction = _find_by_suffix(sample, self.INSTRUCTION_SUFFIXES)
        mask = _find_by_suffix(sample, self.MASK_SUFFIXES)

        if source is None or target is None:
            raise KeyError(
                "ImageEditingDataset samples need a source image and target image "
                f"suffix. Found keys: {sorted(k for k in sample.keys() if not k.startswith('__'))}"
            )

        instruction = _decode_edit_text(instruction)
        if not instruction:
            raise KeyError("ImageEditingDataset samples need an edit instruction.")

        return {
            "source_images": _image_to_tensor(source, self.resolution),
            "target_images": _image_to_tensor(target, self.resolution),
            "edit_instructions": instruction,
            "edit_masks": _mask_to_tensor(mask, self.resolution),
        }

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


if __name__ == '__main__':
    raise SystemExit(0)
