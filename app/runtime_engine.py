import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class RuntimeEngine:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda"
        )

    # def infer(self, prompt: str) -> str:
    #     # infer single prompt

    #     messages = [
    #         {"role": "user", "content": prompt}
    #     ]

    #     formatted_prompt = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )

    #     inputs = self.tokenizer(
    #         formatted_prompt,
    #         return_tensors="pt"
    #     ).to("cuda")
    
    #     with torch.no_grad():
    #         outputs = self.model.generate(
    #             **inputs,
    #             max_new_tokens=50,
    #             temperature=0.7,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.eos_token_id
    #         )
        
    #     return self.tokenizer.decode(
    #         outputs[0][inputs["input_ids"].shape[1]:],
    #         skip_special_tokens=True
    #     )
    
    def infer_batch(self, prompts: list[str]) -> list[str]:
        # infer batch of prompts

        messages_batch = [
            [{"role": "user", "content": prompt}] for prompt in prompts
        ]

        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")

        input_lengths = inputs["attention_mask"].sum(dim=1)
    
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        decoded_outputs = []

        for i, output in enumerate(outputs):
            generated_tokens = output[input_lengths[i]:] # skip the input tokens
            decoded = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            decoded_outputs.append(decoded)
        
        return decoded_outputs