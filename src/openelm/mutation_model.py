import asyncio
import json
import functools
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult
from pydantic import Extra, root_validator
from transformers import BatchEncoding
import transformers

from openelm.codegen import model_setup, set_seed, truncate
from openelm.configs import ModelConfig
from openelm.utils.diff_eval import apply_diff, split_diff
from openelm.inference_hook import vLLMHook
from dotenv import load_dotenv

load_dotenv()

assert "SERVERS_LIST" in os.environ, "Must define SERVERS_LIST in .env"

SERVERS_LIST = os.environ["SERVERS_LIST"]


def get_model(config: ModelConfig):
    if config.model_type == "inference":
        return InferenceServerHuggingFaceLLM(config=config)
    elif config.model_type == "hf":
        return HuggingFaceLLM(config=config)
    elif config.model_type == "openai":
        # Adapt config here
        cfg: dict = {
            "max_tokens": config.gen_max_len,
            "temperature": config.temp,
            "top_p": config.top_p,
            # TODO: rename config option?
            "model_name": config.model_path,
        }
        if "3.5" in config.model_path or "gpt-4" in config.model_path:
            return ChatOpenAI(**cfg)
        else:
            return OpenAI(**cfg)
    elif config.model_type == "hf-reward":
        return HuggingFaceReward(config=config)
    else:
        raise NotImplementedError


class MutationModel(ABC):
    """Base model class for all mutation models."""

    def __init__(self) -> None:
        self.config: ModelConfig

    @abstractmethod
    def generate_programs(self, *args, **kwargs) -> list[str]:
        raise NotImplementedError


class PromptModel(MutationModel):
    """Mutation model that uses prompts to change a seed."""

    def __init__(self, config: ModelConfig) -> None:
        self.config: ModelConfig = config
        seed: int = set_seed(self.config.seed)
        # Use RNG to rotate random seeds during inference.
        self.rng = np.random.default_rng(seed=seed)
        self.model: LLM = get_model(self.config)

    def generate_programs(
        self,
        prompt_dicts: list[dict[str, str]],
        local_scope_truncate: bool,
        do_trunc=True,
        **kwargs
    ) -> list[str]:
        """
        Generate new programs from a batch of programs.

        Given a piece of code, do prompt mutation, execute the code,
        and return the result.

        Args:
            prompt_dicts (list[dict[str, str]): A list of dictionaries containing
            the prompt and template for each program.
            local_scope_truncate (bool): Whether or not to truncate the code to
            the local scope.

        Returns:
            A list of code strings.
        """
        prompts = [prompt_dict["prompt"] for prompt_dict in prompt_dicts]
        templates = [prompt_dict["template"] for prompt_dict in prompt_dicts]
        if "3.5" in self.config.model_path or "gpt-4" in self.config.model_path:
            results = []
            for prompt in prompts:
                results.append(self.model.generate([prompt]))
            completions: list[str] = [
                llmresult.generations[0][0].text for llmresult in results
            ]
        else:
            results = self.model.generate(prompts=prompts)
            completions = [
                gen.text for sublist in results.generations for gen in sublist
            ]
        # Flatten nested list of generations

        if do_trunc:
            trunc = functools.partial(truncate, only_local_scope=local_scope_truncate)
            truncations: list[str] = [
                templates[i] + trunc(completions[i]) for i in range(len(completions))
            ]
        else:
            truncations: list[str] = [
                templates[i] + "\n    " + completions[i]
                for i in range(len(completions))
            ]

        return truncations


class DiffModel(PromptModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def generate_programs(
        self, prompt_dicts: list[dict[str, str]], local_scope_truncate: bool, **kwargs
    ) -> list[str]:
        # local_scope_truncate = False
        prompts = [prompt_dict["prompt"] for prompt_dict in prompt_dicts]
        templates = [prompt_dict["template"] for prompt_dict in prompt_dicts]
        results: LLMResult = self.model.generate(prompts=prompts)
        # Flatten nested list of generations
        completions: list[str] = [
            gen.text for sublist in results.generations for gen in sublist
        ]

        end_of_diff = re.compile("\n[^ +-@]+")
        trunc = functools.partial(truncate, only_local_scope=local_scope_truncate)
        truncations: list[str] = [
            templates[i] + trunc(completions[i]) for i in range(len(completions))
        ]
        outputs: list[str] = []
        for i, code in enumerate(truncations):
            # split the diff text according to <NME>, <BEF>, <MSG>, <DFF>.
            parsed: dict = split_diff(code)
            # truncate the diff hunk at the first line not starting with " ",
            # "+", "-", or "@".
            if parsed and all(
                (s in parsed for s in ["name", "file", "message", "diff"])
            ):
                diff_hunk: str = end_of_diff.split(parsed["diff"])[0]
                nme_idx: int = diff_hunk.find("<NME>")
                if nme_idx != -1:
                    diff_hunk = diff_hunk[:nme_idx]
                outputs.append(apply_diff(prompts[i], diff_hunk))
        return outputs

class HuggingFaceReward:

    def __init__(self, config: ModelConfig):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_path)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()


    def __call__(self, prompt: str) -> torch.Tensor:
        """Compute the reward for the prompt."""
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            encodings = encodings.cuda()
        with torch.inference_mode():
            logits = self.model(**encodings).logits
        return logits

sema = asyncio.BoundedSemaphore(10)

class InferenceServerHuggingFaceLLM:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = vLLMHook(
                model_path=config.model_path,
                tensor_parallel_size=1,
                servers=json.loads(os.environ["SERVERS_LIST"]),
                sems=5,
            )

    def __call__(
        self, prompt: str
    ) -> str:
        async def get_results(prompt):
            async with sema:
                texts = await self.model.generate(prompts=[prompt], max_new_tokens=256, temperature=1.0)
                return texts
        # This portion needs to be modified to handle the coroutine and get the result
        texts = asyncio.run(get_results(prompt))

        # generations = [Generation(text=text["outputs"][0]) for text in texts]
        print(f"prompt: {prompt}")
        print(f"result: {texts[0]['outputs'][0]}")
        return texts[0]["outputs"][0]

    def generate(
        self, prompts: list[str], stop: Optional[list[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        async def get_results(prompts):
            async with sema:
                texts = await self.model.generate(prompts=prompts, max_new_tokens=256, temperature=1.0)
                return texts
        # This portion needs to be modified to handle the coroutine and get the result
        texts = asyncio.run(get_results(prompts))

        print(f"prompt: {prompt}")
        print(f"result: {texts[0]['outputs'][0]}")
        
        generations = [Generation(text=text["outputs"][0]) for text in texts]

        generations_dict = {}
        for j, prompt in enumerate(prompts):
            generations_dict[prompt].extend(generations[j])

        # import ipdb; ipdb.set_trace()
        return LLMResult(generations=list(generations_dict.values()))


class HuggingFaceLLM:#(LLM):
    config: ModelConfig
    model: Any = None
    tokenizer: Any = None
    device: Any = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow

    def __init__(self, config: ModelConfig):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_path)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "huggingface"

    # def _call(self, prompt: str, stop: Optional[list[str]] = None):
    def __call__(self, prompts: str, stop: Optional[list[str]] = None):
        """Run the LLM on the given prompt and input."""
        encodings = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**encodings)
            logits = outputs.logits
        return logits

    def generate(
        self, prompts: list[str], stop: Optional[list[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        batch_size = self.config.batch_size
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        encodings = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generations_dict: dict[str, list[Generation]] = defaultdict(list)

        for i in range(total_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(prompts))
            batched_prompts = BatchEncoding(
                {
                    "input_ids": encodings["input_ids"][start_index:end_index],
                    "attention_mask": encodings["attention_mask"][
                        start_index:end_index
                    ],
                }
            ).to(self.device)
            if self.config.logits_only:
                with torch.inference_mode():
                    outputs = self.model(**batched_prompts)
                    if i == 0:
                        logits = outputs.logits
                    else:
                        logits = torch.cat((logits, outputs.logits), dim=0)
                generations: list[Generation] = [
                    Generation(text="", generation_info={"logits": logits})
                    for logits in logits
                ]
            else:
                input_ids_len: int = batched_prompts["input_ids"].shape[1]
                with torch.inference_mode():
                    tokens = self.model.generate(
                        **batched_prompts,
                        do_sample=self.config.do_sample,
                        num_return_sequences=self.config.num_return_sequences,
                        temperature=self.config.temp,
                        max_new_tokens=self.config.gen_max_len,
                        top_p=self.config.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    texts: list[str] = self.tokenizer.batch_decode(
                        tokens[:, input_ids_len:, ...]
                    )
                generations = [Generation(text=text) for text in texts]

            for j, prompt in enumerate(prompts[start_index:end_index]):
                slice_start = j * self.config.num_return_sequences
                slice_end = slice_start + self.config.num_return_sequences
                generations_dict[prompt].extend(generations[slice_start:slice_end])

        return LLMResult(generations=list(generations_dict.values()))
