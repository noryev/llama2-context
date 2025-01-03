from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import os
import logging
import argparse

def read_context_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def format_prompt(question, context):
    return f"""Based on this text, answer the question in one clear paragraph. Only use information directly stated in the text.

Text: {context}

Question: {question}

Short Answer:"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question', type=str, nargs='+')
    parser.add_argument('--context', type=str)
    
    args = parser.parse_args()
    question = ' '.join(args.question)
    print(f"Question: {question}\n")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "/app/model",
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("/app/model")
    
    # Get context
    context = read_context_file(args.context) if args.context else None
    
    # Generate response
    prompt = format_prompt(question, context)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.1,
        do_sample=False
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the answer part
    if "Short Answer:" in response:
        response = response.split("Short Answer:")[-1].strip()
    
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
