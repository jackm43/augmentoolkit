import asyncio
import uuid
from openai import AsyncOpenAI
import cohere
from httpx import Timeout
import json

def make_id():
    return str(uuid.uuid4())


class EngineWrapper:
    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
        mode="api",  # can be one of api, aphrodite, llama.cpp, cohere
        quantization="gptq",  # only needed if using aphrodite mode
    ):
        self.mode = mode
        self.model = model
        try:
            if mode == "cohere":
                self.client = cohere.AsyncClient(api_key=api_key)
            elif mode == "api":
                if base_url and not api_key:
                    # For local endpoints, use httpx directly
                    import httpx
                    self.client = httpx.AsyncClient(
                        base_url=base_url,
                        timeout=Timeout(timeout=5000.0, connect=10.0)
                    )
                    self.is_local = True
                else:
                    # For OpenAI API, use their client
                    self.client = AsyncOpenAI(
                        timeout=Timeout(timeout=5000.0, connect=10.0),
                        api_key=api_key,
                        base_url=base_url
                    )
                    self.is_local = False
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize client: {str(e)}")

    async def submit_completion(
        self, prompt, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        if "n_predict" not in sampling_params:
            sampling_params["n_predict"] = sampling_params["max_tokens"]
        
        use_min_p = False
        if "min_p" in sampling_params:
            use_min_p = True

        if self.mode == "api":
            timed_out = False
            completion = ""
            try:
                if self.is_local:
                    # Direct API call for local endpoints
                    payload = {
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": sampling_params["temperature"],
                        "top_p": sampling_params["top_p"],
                        "stop": sampling_params["stop"],
                        "max_tokens": sampling_params["max_tokens"],
                        "stream": True
                    }
                    if use_min_p:
                        payload["min_p"] = sampling_params["min_p"]

                    print("SENDIN PAYLOAD")
                    print(payload)
                    
                    async with self.client.stream("POST", "/v1/completions", json=payload) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    chunk = json.loads(line[6:])
                                    if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                                        completion += chunk["choices"][0]["delta"]["content"]
                                except json.JSONDecodeError:
                                    continue
                else:
                    # Use OpenAI client for API endpoints
                    completion_kwargs = {
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": sampling_params["temperature"],
                        "top_p": sampling_params["top_p"],
                        "stop": sampling_params["stop"],
                        "max_tokens": sampling_params["max_tokens"],
                        "stream": True,
                        "timeout": 360,
                    }
                    if use_min_p:
                        completion_kwargs["extra_body"] = {"min_p": sampling_params["min_p"]}
                    
                    stream = await self.client.completions.create(**completion_kwargs)
                    async for chunk in stream:
                        try:
                            completion = completion + chunk.choices[0].delta.content
                        except Exception as e:
                            raise ConnectionError(f"Failed to process stream chunk: {str(e)}")
            except Exception as e:
                raise ConnectionError(f"Failed to create completion stream: {str(e)}")

            return prompt + completion, timed_out

        if self.mode == "cohere":
            raise Exception("Cohere not compatible with completion mode!")

    async def submit_chat(
        self, messages, sampling_params
    ):  # Submit request and wait for it to stream back fully
        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 3000
        if "stop" not in sampling_params:
            sampling_params["stop"] = []
        
        use_min_p = False
        if "min_p" in sampling_params:
            use_min_p = True

        if self.mode == "api":
            completion = ""
            timed_out = False
            try:
                if self.is_local:
                    # Direct API call for local endpoints
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": sampling_params["temperature"],
                        "top_p": sampling_params["top_p"],
                        "stop": sampling_params["stop"],
                        "max_tokens": sampling_params["max_tokens"],
                        "stream": True
                    }
                    if use_min_p:
                        payload["min_p"] = sampling_params["min_p"]
                    
                    async with self.client.stream("POST", "/v1/chat/completions", json=payload) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    chunk = json.loads(line[6:])
                                    if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                                        completion += chunk["choices"][0]["delta"]["content"]
                                except json.JSONDecodeError:
                                    continue
                else:
                    # Use OpenAI client for API endpoints
                    chat_kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": sampling_params["temperature"],
                        "top_p": sampling_params["top_p"],
                        "stop": sampling_params["stop"],
                        "max_tokens": sampling_params["max_tokens"],
                        "stream": True,
                    }
                    if use_min_p:
                        chat_kwargs["extra_body"] = {"min_p": sampling_params["min_p"]}
                    
                    stream = await self.client.chat.completions.create(**chat_kwargs)
                    async for chunk in stream:
                        try:
                            if chunk.choices[0].delta.content:
                                completion = completion + chunk.choices[0].delta.content
                        except Exception as e:
                            raise ConnectionError(f"Failed to process chat stream chunk: {str(e)}")
            except Exception as e:
                raise ConnectionError(f"Failed to create chat stream: {str(e)}")

            return completion, timed_out

        elif self.mode == "cohere":
            timed_out = False
            completion = ""
            try:
                messages_cohereified = [
                    {
                        "role": "USER" if message["role"] == "user" else "CHATBOT",
                        "message": message["content"],
                    }
                    for message in messages
                ]
                stream = self.client.chat_stream(
                    model=self.model,
                    chat_history=messages_cohereified[1:-1],
                    message=messages_cohereified[-1]["message"],
                    preamble=messages_cohereified[0]["message"],
                    temperature=sampling_params["temperature"],
                    p=sampling_params["top_p"],
                    stop_sequences=sampling_params["stop"],
                    max_tokens=sampling_params["max_tokens"],
                )
                async for chunk in stream:
                    try:
                        if chunk.event_type == "text-generation":
                            completion = completion + chunk.text
                    except Exception as e:
                        raise ConnectionError(f"Failed to process Cohere stream chunk: {str(e)}")
            except Exception as e:
                raise ConnectionError(f"Failed to create Cohere chat stream: {str(e)}")

            return completion, timed_out

        else:
            raise Exception("Aphrodite not compatible with chat mode!")