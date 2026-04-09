from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from easyroutine.interpretability import HookedModel
from rich import print
import io
import base64
from PIL import Image
import json
import re
import datetime
from tqdm import tqdm
import ollama
from typing import Literal
from src.utils import get_whoops_element_by_id, start_ollama, ollama_model_map

load_dotenv(".env")

# data = data["saved_data"]
WHOOPS = load_dataset("nlphuji/whoops", split="test")

SYSTEM_PROMPT = """
You are presented with an image and an incomplete sentence describing its content. The image intentionally portrays an unusual scenario that contrasts typical or factual knowledge.

Your task is to generate two lists of tokens:

    1. Factual Tokens (5 tokens): These tokens should represent words or concepts that accurately and typically complete the sentence based solely on common knowledge, without considering the unusual image.

    2. Counterfactual Tokens (5 tokens): These tokens should represent words or concepts that correctly complete the sentence when explicitly considering the unusual content depicted in the image, even if it contradicts common factual knowledge.

Please format your response clearly as a JSON object as follows:

```json
{
  "sentence": "{INCOMPLETE_SENTENCE}",
  "factual_tokens": ["token1", "token2", "token3", "token4", "token5"],
  "counterfactual_tokens": ["token1", "token2", "token3", "token4", "token5"]
}
```

Choose tokens that clearly differentiate between typical knowledge and the unusual scenario depicted by the provided image.
"""

SYSTEM_PROMPT_NO_IMAGE = """
You are presented with an incomplete sentence describing a scenario. You do not have access to any visual information (no image is provided). Your task is to generate a list of 5 tokens that factually and accurately complete the sentence based solely on general knowledge and common sense.

Please format your response clearly as a JSON object as follows:
```json
{
    "sentence": "{INCOMPLETE_SENTENCE}",
    "factual_tokens": ["token1", "token2", "token3", "token4", "token5"]
}
```
Use the same key names and structure as the json object above, and replace the placeholder text with your response.
"""

SYSTEM_PROMPT_WITH_IMAGE = """
You are presented with an image and an incomplete sentence describing its content. The image intentionally portrays an unusual scenario that contrasts typical or factual knowledge. Your task is to generate a list of 5 tokens that complete the sentence based on the unusual content depicted in the image and that would NOT be expected based on common knowledge.

Please format your response clearly as a JSON object as follows:
```json
{
    "sentence": "{INCOMPLETE_SENTENCE}",
    "counterfactual_tokens": ["token1", "token2", "token3", "token4", "token5"]
}
```
Use the same key names and structure as the json object above, and replace the placeholder text with your response.
"""

PROMPT = """
Text: {text}
"""

def parse_response(response):
    # Extract JSON string from longer text response
    json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
    if json_match:
        json_string = json_match.group(1).strip()
    else:
        # try to extract JSON string without code block (remove all the new lines, strip and take the text between the curly brackets)
        json_string = re.search(r'{(.*)}', response, re.DOTALL).group(0).strip()
        if not json_string:
            raise ValueError("JSON format not found in the provided text.")

    # Parse the JSON string into a Python dictionary
    token_dict = json.loads(json_string)

    return token_dict

def generate_target_token_using_target_model(
    data_file, target_model
    ): 
    new_data = []
    
        # save the data
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    hours = now.strftime("%H-%M-%S")
    output_filename = f"./data/openai/visual_counterfactual_with_llava_factual_tokens_{date}_{hours}.json"
    
    start_ollama(model_name = target_model)
    
    data = json.load(open(data_file))
    print("Data loaded. Total number of data points: ", len(data))
    
    for index in tqdm(range(len(data))):
        image = get_whoops_element_by_id(WHOOPS, data[index]["image_id"])["image"] # type: ignore
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")  # Use "PNG" if preferred
        image_bytes = buffered.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        text = data[index]["text"]
        
        try:


            parsed_response = parse_response(response.message.content)
            
            # could be that the second key is not "factual_tokens" but the model create a different key, so we need to check and change it
            if "factual_tokens" not in parsed_response:
                factual_tokens = parsed_response[list(parsed_response.keys())[1]]
            else:
                factual_tokens = parsed_response["factual_tokens"]
            
            new_data.append(data[index])
            new_data[-1]["llava7_factual_tokens"] = factual_tokens  
            
            
            ## counterfactual
            response = ollama.chat(
                model=ollama_model_map[target_model],
                messages=[
                    {"role": "user", "content": SYSTEM_PROMPT_WITH_IMAGE},
                    {"role": "user", "content": "Text: {text}".format(text=text)}
                    # {"role": "user", "images": [base64_image]}
                ]
            )
            parsed_response = parse_response(response.message.content)
            if "counterfactual_tokens" not in parsed_response:
                counterfactual_tokens = parsed_response[list(parsed_response.keys())[1]]
            else:
                counterfactual_tokens = parsed_response["counterfactual_tokens"]
                
            new_data[-1]["llava7_counterfactual_tokens"] = counterfactual_tokens
            
            # save the data
            with open(output_filename, "w") as f:
                json.dump(new_data, f, indent=4)
        except Exception as e:
            print("Error: ", e)
            continue

if __name__ == "__main__":

    generate_target_token_using_target_model(
        data_file= "data/openai/visual_counterfactual_2025-03-17_18-05-09.json",
        target_model="llava-7b"
    )