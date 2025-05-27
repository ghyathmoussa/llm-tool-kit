


def format_reward_func(completions, **kwargs):
    # Example: Check if the completion follows a specific <think>...</think><answer>...</answer> format
    # This is a placeholder. Implement your actual logic.
    # See: https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function
    rewards = []
    for comp_list in completions: # completions is a list of lists of dicts
        # Assuming conversational format [{ "role": "assistant", "content": "..." }]
        if comp_list and isinstance(comp_list, list) and len(comp_list) > 0 and "content" in comp_list[0]:
            content = comp_list[0]["content"]
            # A very basic check
            if "<think>" in content and "</think>" in content and "<answer>" in content and "</answer>" in content:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0) # Penalize malformed completions
    return rewards

def accuracy_reward_func(completions, solution, **kwargs):
    # Example: Check if the answer in the completion matches the provided solution
    # This is a placeholder. Implement your actual logic.
    # 'solution' would come from your dataset if you pass `remove_unused_columns=False`
    # and your dataset has a 'solution' column.
    rewards = []
    if not solution or len(solution) != len(completions):
        return [0.0] * len(completions)

    for i, comp_list in enumerate(completions):
        if comp_list and isinstance(comp_list, list) and len(comp_list) > 0 and "content" in comp_list[0]:
            generated_content = comp_list[0]["content"]
            # Placeholder: extract answer from generated_content and compare with solution[i]
            # current_solution = solution[i] 
            # extracted_answer = ... parse generated_content ...
            # if extracted_answer == current_solution:
            #     rewards.append(1.0)
            # else:
            #     rewards.append(0.0)
            rewards.append(0.0) 
        else:
            rewards.append(0.0) 
    return rewards