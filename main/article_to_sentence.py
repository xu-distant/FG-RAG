import torch
import json
import spacy

def generate_propositions(content,model,tokenizer,device):
    title = ""
    section = ""
    input_text = f"Title: {title}. Section: {section}. Content: {content}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids.to(device), max_new_tokens=768).to(device)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        prop_list = json.loads(output_text)
    except:
        prop_list = []
        print("[ERROR] Failed to parse output text as JSON.")

    return prop_list


def generate_propositions_batch(contents, model, tokenizer, device, max_length=768):
    title = ""
    section = ""
    input_texts = [f"Title: {title}. Section: {section}. Content: {content}" for content in contents]
    if not input_texts:
        print("[ERROR] No input texts provided.")
        return []
    # Use a tokenizer to process the input, ensuring it does not exceed the maximum length.
    input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=max_length).input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=768)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    all_propositions = []
    for output_text in output_texts:
        try:
            prop_list = json.loads(output_text)
            all_propositions.extend(prop_list)
        except json.JSONDecodeError:
            print("[ERROR] Failed to parse output text as JSON.")

    return all_propositions