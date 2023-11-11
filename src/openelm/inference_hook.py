import asyncio
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import aiohttp
import torch


semaphore = asyncio.Semaphore(5)


class InferenceHook(ABC):
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (`Dict[str, Any]`): a dictionary of parameters to initilize the model with
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self, prompts: List[str], **kwargs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Args:
            prompts (`Union[List[str], ST]`): inputs for generations
            kwargs (`Dict[str, Any]`): parameters to control generation

        Returns:
            outputs (`List[Dict[str, Any]]`): a list of dictionaries, each dictionary contains the following keys:
                id (`int`): the id of the prompt
                prompt (`str`): the prompt
                outputs (`ST`): a list of outputs per prompt
        """
        raise NotImplementedError

    def clean(self, generations, **kwargs) -> List[str]:
        """
        Args:
            generations (`Any`): the output of `generate`
            kwargs (`Dict[str, Any]`): parameters to control generation
        Returns:
            cleaned (`List[str]`): a list of cleaned outputs
        """

        raise NotImplementedError

    @abstractmethod
    def free(self):
        """
        Clean up resources after the inference
        """
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.generate(*args, **kwds)
    
    def __repr__(self) -> str:
        """
        Returns:
            `str`: a string representation of the hook
        """
        raise NotImplementedError


class vLLMHook(InferenceHook):
    def __init__(
        self,
        model_path,
        tensor_parallel_size=1,
        num_external_nodes=0,
        servers=None,
        sems=5,
    ):
        """
        Starts data parallel vLLM servers either locally or on separate nodes by spawning slurm jobs

        Args:
            model_path (`str`): the path to the model
            tensor_parallel_size (`int`): the number of GPUs to use per one server
            num_external_nodes (`int`): spawn this many slurm jobs for the servers, if `0`, use only local resourses
        """
        self.init_time = time.time()
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.num_external_nodes = num_external_nodes
        self.nth_request = 0

        devices = list(map(str, range(torch.cuda.device_count())))
        devices = [
            ",".join(devices[i * tensor_parallel_size : (i + 1) * tensor_parallel_size])
            for i in range(len(devices) // tensor_parallel_size)
        ]
        if servers is None:
            if num_external_nodes:
                self.job_ids = []
                self.servers = []
                self.data_parallel_size = (
                    torch.cuda.device_count()
                    * num_external_nodes
                    // tensor_parallel_size
                )

                sbatch_script_path = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "vllm.sbatch"
                )
                for _ in range(num_external_nodes):
                    cmd = f"sbatch {sbatch_script_path} NUM_TP={tensor_parallel_size} MODEL_PATH={model_path} DEVICES={'|'.join(devices)}"
                    process = subprocess.Popen(
                        cmd.split(),
                        stdout=subprocess.PIPE,
                        env={**os.environ, "TORCHELASTIC_USE_AGENT_STORE": ""},
                    )

                    while True:
                        output = process.stdout.readline().decode("utf-8").strip()
                        if output == "" and process.poll() is not None:
                            break
                        if output:
                            print(output)
                            if output.startswith("Submitted batch job"):
                                self.job_ids.append(output.split()[-1].strip())

                    while not os.path.exists(f"{self.job_ids[-1]}"):
                        time.sleep(1)

                    with open(f"{self.job_ids[-1]}") as log:
                        while True:
                            output = log.readline().strip()
                            if output:
                                print(output)
                                if output.startswith("HOSTNAME="):
                                    hostname = output.split("=")[-1].strip()
                                    self.servers.extend(
                                        [
                                            f"{hostname}:{8000+i}"
                                            for i in range(8 // tensor_parallel_size)
                                        ]
                                    )
                                    break

            else:
                self.data_parallel_size = (
                    torch.cuda.device_count() // tensor_parallel_size
                )
                self.servers = [
                    f"localhost:{8000+i}" for i in range(self.data_parallel_size)
                ]
                self.processes = []
                for i in range(self.data_parallel_size):
                    cmd = f"python -m vllm.entrypoints.api_server -tp={tensor_parallel_size} --dtype=float16 --model={model_path} --port {8000+i}"
                    kwargs = {
                        "env": {
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": devices[i],
                            "TORCHELASTIC_USE_AGENT_STORE": "",
                        }
                    }
                    if not os.environ.get("DEBUG", False):
                        print("Debug mode disabled!")

                    log_file = open(
                        f"process_{i}.log", "w"
                    )  # Open a file for writing logs
                    error_file = open(
                        f"process_{i}.err", "w"
                    )  # Open a file for writing errors

                    kwargs["stdout"] = log_file
                    kwargs["stderr"] = error_file

                    process = subprocess.Popen(cmd.split(), **kwargs)
                    self.processes.append(process)

                print(
                    f"Loading {self.data_parallel_size} processes for {model_path}..."
                )

            not_loaded = list(self.servers)
            while not_loaded:
                for server in not_loaded:
                    try:
                        asyncio.run(
                            self.request_vllm_api(
                                server=server, prompt=".", max_new_tokens=1
                            )
                        )
                        not_loaded.remove(server)
                    except aiohttp.client_exceptions.ClientConnectorError:
                        break

                time.sleep(1)

            self.load_time = time.time() - self.init_time
            print(f"Loaded {model_path} in {self.load_time:.0f}s")
        else:
            self.processes = []
            self.servers = servers
            self.data_parallel_size = len(servers)
            self.load_time = 0

    async def request_vllm_api(
        self,
        prompt: str,
        i=0,
        best_of=1,
        top_p=1.0,
        top_k=-1,
        num_return_sequences=1,
        temperature=0.0,
        max_new_tokens=512,
        stop=[],
        server=None,
    ):
        pload = {
            "prompt": prompt,
            "best_of": best_of,
            "top_p": top_p,
            "top_k": top_k,
            "n": num_return_sequences,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stop": stop,
            "stream": False,
        }

        if server is None:
            server = self.servers[self.nth_request % self.data_parallel_size]
            self.nth_request += 1

        connector = aiohttp.TCPConnector(limit_per_host=32768)
        timeout = aiohttp.ClientTimeout(total=3600)
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            async with session.post(
                f"http://{server}/generate", json=pload
            ) as response:
                try:
                    data = await response.json()
                    return {
                        "id": i,
                        "prompt": prompt,
                        "outputs": [x[len(prompt) :] for x in data["text"]],
                    }
                except aiohttp.client.ContentTypeError:
                    return {"id": i, "prompt": prompt, "outputs": [None]}

    async def generate(
        self, prompts: List[str], **kwargs
    ) -> List[Dict[str, Any]]:

        async def generate_vllm_api(prompts, **kwargs):
            outputs = [
                self.request_vllm_api(prompt=prompt, i=i, **kwargs)
                for i, prompt in enumerate(prompts)
            ]
            return await asyncio.gather(*outputs)

        batch_size = 32768
        outputs = []
        for i in range(0, len(prompts), batch_size):
            outputs += await generate_vllm_api(prompts[i : i + batch_size], **kwargs)

        return outputs

    def free(self):
        if self.num_external_nodes:
            if self.job_ids:
                subprocess.run(f"scancel {' '.join(self.job_ids)}".split())
                self.job_ids = []
        else:
            for p in self.processes:
                os.kill(p.pid, signal.SIGTERM)
                p.communicate()
            print(f"Unloaded all {self.model_path} processes")
            self.processes = []

    def clean(self, generations, **kwargs) -> List[str]:
        return [x["outputs"][0] for x in generations]

    def __del__(self):
        self.free()
    
    def __repr__(self) -> str:
        # we should include information about the model used, the server and that it is a vLLM hook
        return f"vLLMHook(model_path={self.model_path}, servers={self.servers}, tensor_parallel_size={self.tensor_parallel_size}, num_external_nodes={self.num_external_nodes})"
