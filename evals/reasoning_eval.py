import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PeftModel

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class EvaluateModel:
    def __init__(self, model_name: str, output_dir: str = None): # Added output_dir for PEFT model
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.padding_side = "right" # Defined in tokenizer loading

    def _load_tokenizer(self):
        # If evaluating a PEFT model, tokenizer should be from base model
        # If output_dir is provided, assume it's a PEFT model, load base tokenizer
        # Otherwise, load from model_name (could be a fully merged model)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name) # Try base model first
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir) # Fallback to output_dir if base fails

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # For generation, left padding is usually preferred
        return tokenizer
    
    def evaluate(self,
                 test_prompt_text: str, # Made prompt an argument
                 temperature: float = 0.3,
                 max_new_tokens: int = 512,
                 top_p: float = 0.9,
                 do_sample: bool = True
            ):
        
        if self.output_dir: # PEFT model evaluation
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, # Base model name
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16, # Ensure consistent dtype
            )
            model_to_eval = PeftModel.from_pretrained(base_model, self.output_dir)
            model_to_eval.eval() # Set to evaluation mode
        else: # Evaluating a fully merged model or base model without adapter
            model_to_eval = AutoModelForCausalLM.from_pretrained(
                self.model_name, # This could be output_dir if model was saved fully
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            model_to_eval.eval()

        tokenizer = self._load_tokenizer()
        
        # Example test prompt (Arabic), user should pass their own
        # test_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

        # ### Instruction:
        # قم بحل المعادلة التالية خطوة بخطوة: 3x² + 7x - 10 = 0

        # ### Response:
        # """
        # Construct prompt in conversational format if model expects it
        # For GRPO fine-tuned models, it likely expects the system prompt + user prompt
        eval_prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_prompt_text}
        ]
        
        # Apply chat template if available and appropriate for the model
        try:
            final_prompt_str = tokenizer.apply_chat_template(eval_prompt_messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Could not apply chat template: {e}. Using simple concatenation.")
            final_prompt_str = SYSTEM_PROMPT + "\nUser: " + test_prompt_text + "\nAssistant:"


        inputs = tokenizer(final_prompt_str, return_tensors="pt").to(self.device)
        
        print(f"Generating response for: {final_prompt_str}")

        with torch.no_grad(): # Ensure no gradients are computed during generation
            outputs = model_to_eval.generate(
                **inputs, # Pass all inputs from tokenizer
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id # Important for open-ended generation
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("--- Model Output ---")
        print(decoded_output)
        print("--------------------")
        # To get only the generated part:
        # generated_part = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        # print("--- Generated Part Only ---")
        # print(generated_part)
        # print("-------------------------")

