import re
import os
import traceback
import logging
import yaml
from augmentoolkit.generation_functions.safe_formatter import safe_format


class GenerationStep:
    def __init__(
        self,
        prompt_path="",  # relative to the Inputs directory
        regex=re.compile(r".*", re.DOTALL),  # take whole completion
        sampling_params={
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 3000,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "### Information",
                "## Information",
                "## Instruction",
                "Name:",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
        },
        completion_mode=True,  # Chat vs completion mode
        retries=0,
        engine_wrapper=None,
        logging_level=logging.INFO,  # Default logging level
        output_processor=lambda x: x,  # to ensure that control flow does not need to have decision code handling the outputs of the LLM, you can pass in a function to handle and modify the outputs (post regex) here. By default it's just the identity function and does nothing.
        return_input_too=True,
        default_prompt_folder="prompts",
        prompt_folder="prompts",
        use_stop=True,
    ):
        self.prompt_path = prompt_path
        self.regex = regex
        self.sampling_params = sampling_params
        if not use_stop:
            del self.sampling_params["stop"]
        self.completion_mode = completion_mode
        self.retries = retries
        self.logging_level = logging_level
        self.output_processor = output_processor
        self.return_input_too = return_input_too
        if not engine_wrapper:
            raise Exception("Engine wrapper not passed in!")
        self.engine_wrapper = engine_wrapper
        self.prompt_folder = prompt_folder
        self.default_prompt_folder = default_prompt_folder
        logging.basicConfig(
            level=self.logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    async def generate(self, **kwargs):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ideal_path = os.path.join(
                current_dir, "..", "..", self.prompt_folder, self.prompt_path
            )
            full_prompt_path = ideal_path if os.path.exists(ideal_path) else os.path.join(
                current_dir, "..", "..", self.default_prompt_folder, self.prompt_path
            )

            try:
                with open(full_prompt_path, "r", encoding='utf-8') as pf:
                    prompt = pf.read()
            except FileNotFoundError:
                raise Exception(f"Prompt file not found at {full_prompt_path}")
            except Exception as e:
                raise Exception(f"Error reading prompt file: {str(e)}")

            times_tried = 0
            response = None  # Initialize response variable
            if self.completion_mode:
                response = await self._handle_completion_mode(prompt, times_tried, **kwargs)
            else:
                response = await self._handle_chat_mode(prompt, times_tried, **kwargs)
            return response

        except Exception as e:
            logging.error(f"Fatal error in generate(): {str(e)}")
            raise

    async def _handle_completion_mode(self, prompt, times_tried, **kwargs):
        prompt_formatted = safe_format(prompt, **kwargs)
        while times_tried <= self.retries:
            try:
                response, timeout = await self.engine_wrapper.submit_completion(
                    prompt_formatted, self.sampling_params
                )
                filtered_response = re.search(self.regex, response).group(1)
                ret = self.output_processor(filtered_response)
                return (ret, prompt_formatted + filtered_response) if self.return_input_too else (ret, timeout)
            except Exception as e:
                self._log_completion_error(e, response)
                times_tried += 1
        raise Exception("Generation step failed -- too many retries!")

    async def _handle_chat_mode(self, prompt, times_tried, **kwargs):
        try:
            messages = yaml.safe_load(prompt)
        except yaml.YAMLError as e:
            raise Exception(f"Invalid YAML in prompt file: {str(e)}")

        messages = self._format_messages(messages, **kwargs)
        
        while times_tried <= self.retries:
            try:
                messages = [{"role": m["role"], "content": m["content"].strip()} for m in messages]
                response, timeout = await self.engine_wrapper.submit_chat(
                    messages, self.sampling_params
                )
                ret = self.output_processor(response)
                if self.return_input_too:
                    return ret, yaml.dump(
                        messages + [{"role": "assistant", "content": response, "timeout": timeout}],
                        default_flow_style=False,
                        allow_unicode=True
                    )
                return ret, timeout
            except Exception as e:
                self._log_chat_error(e, messages, response)
                times_tried += 1
        raise Exception("Generation step failed -- too many retries!")

    def _format_messages(self, messages, **kwargs):
        new_messages = []
        for message in messages['messages']:
            try:
                new_messages.append({
                    "role": message["role"],
                    "content": safe_format(message["content"], **kwargs)
                })
            except Exception as e:
                raise Exception(f"Failed to format message: {str(e)}")
        return new_messages

    def _log_completion_error(self, error, response):
        logging.error(f"Error in completion mode: {str(error)}")
        if not self.engine_wrapper.mode == "llamacpp":
            logging.error(f"Response: {response}")
        raise error

    def _log_chat_error(self, error, messages, response):
        logging.error(f"Error in chat mode: {str(error)}")
        logging.error(f"Messages: {yaml.dump(messages, default_flow_style=False, allow_unicode=True)}")
        try:
            logging.error(f"Response: {response}")
        except UnboundLocalError:
            logging.error("No response available")
        raise error
