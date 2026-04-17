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
from src.utils import start_ollama

load_dotenv(".env")

# Updated system prompt with JSON output format instructions.
SYSTEM_PROMPT = """
You are an helpfull assistant expert in LLMs research. 

Counterfactual Dataset Generation Prompt

Objective:
Generate captions for images that highlight a clear contrast between common (factual) and unusual (counterfactual) scenarios involving the subject depicted. Each caption must include the subject of the image and end with "___" indicating the blank space where a single-word token is placed.

Definitions:
- **Factual token**: A single word that represents typical, expected behavior or attributes of the main subject shown in the image.
- **Counterfactual token**: A single word introducing a surprising, unexpected, or unusual element related explicitly to the same main subject; it makes sense only if the image explicitly illustrates this twist.

Context Provided:
For each image, you will receive the following textual information:
- Selected Caption: A primary description identifying the main subject clearly.
- Crowd Captions: Alternative descriptions from multiple annotators.
- Designer Explanation: Explanation emphasizing the unusual or counterintuitive aspect involving the subject.
- Crowd Explanations: Multiple explanations focusing on the unusual aspects related directly to the subject of the image.

Task Instructions:

Caption Construction:
- Create exactly one neutral sentence (caption) clearly containing the main subject depicted in the image but avoiding the description of unusual aspect contained in the image.
- The sentence must end with an intentional blank ("___").
- Critical Requirement: The caption must compel the model to complete the blank differently based on the context:
    - **Without the image**: complete with a factual token (typical scenario involving the subject).
    - **With the image**: complete with a counterfactual token (unexpected scenario explicitly depicted).
- Important Constraint: Use neutral language with NO textual hints indicating abnormality. The main subject must explicitly appear in the caption to establish context clearly. Only the image content itself should disambiguate the scenario.
- The caption should not contain any unusual or counterintuitive elements; the unusual aspect should be reflected solely in the image content and in the counterfactual tokens.
- Make sure that if you substitute the blank with a factual or counterfactual token, the sentence is fluent and grammatically correct.

Explicit Single-Word Token Generation:
- Generate exactly **ten single-word factual tokens** representing common scenarios involving the main subject that could complete in a grammatically correct way the sentence.
- Generate exactly **ten single-word counterfactual tokens** representing surprising scenarios involving the same subject, justified solely by the provided image and that could complete the sentence in a grammatically correct way.
- Strictly enforce single-word tokens; no multi-word phrases or sentences.
- Ensure clear differentiation without conceptual overlap between factual and counterfactual tokens.

JSON Output Format:
Provide each caption and tokens following this exact schema:

{
  "caption": "Neutral sentence explicitly containing the main subject and ending with an intentional blank ('___')",
  "factual_tokens": ["token1", "token2", "token3", "token4", "token5", ...],
  "counterfactual_tokens": ["token1", "token2", "token3", "token4", "token5", ...],
  "context": {
    "selected_caption": "Primary description clearly stating the main subject of the image",
    "crowd_captions": ["Caption 1", "Caption 2", "..."],
    "designer_explanation": "Explanation highlighting the unusual aspect directly involving the main subject",
    "crowd_explanations": ["Explanation 1", "Explanation 2", "..."]
  }
}

Your role is to craft neutral captions explicitly containing the main subject of each image, along with precisely differentiated factual and counterfactual single-word tokens. The explicit presence of the main subject in the caption must guide factual versus counterfactual completions, relying solely on the provided image for disambiguation.
"""




# Define additional prompt constants
BASE_PROMPT_BEFORE_IMG = (
    "Please generate candidate captions according to the instructions above. You shouls output a JSON string. Image and context:"
)
BASE_PROMPT_AFTER_IMG = (
    "Here is the context for the image:\n"
    "Selected Caption: {selected_caption}\n"
    "Crowd Captions:\n{crowd_captions_itemized}\n"
    "Designer Explanation: {designer_explanation}\n"
    "Crowd Explanations:\n{crowd_explanation_itemized}\n\n"
    "Now, generate a caption with an intentional blank at the end following the instructions."
)

CLIENT = OpenAI()




def get_model_response(prompt, image, model: Literal["openai", "llava-7b", "llava-34b"]):
    if model == "openai":
        response = CLIENT.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": BASE_PROMPT_BEFORE_IMG},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        if not response.choices:
            raise ValueError("No response choices were returned.")
        if not response.choices[0].message.content:
            raise ValueError("No message was returned in the response.")
        response = response.choices[0].message.content
    elif model == "llava-7b":
        response = ollama.chat(
            model="llava:7b-v1.6-mistral-fp16",
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT},
                {"role": "user", "content": BASE_PROMPT_BEFORE_IMG},
                {"role": "user", "images": [image]},
                {"role": "user", "content": prompt},
            ],
        )
        response = response.message.content
        # remove ale \n in the response
        response = response.replace("\n", "").strip()
    elif model == "llava-34b":
        response = ollama.chat(
            model="llava:34b",
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT},
                {"role": "user", "content": BASE_PROMPT_BEFORE_IMG},
                {"role": "user", "images": [image]},
                {"role": "user", "content": prompt},
            ],
        )
        response = response.message.content
        # remove ale \n in the response
        response = response.replace("\n", "").strip()
    elif model == "llama-3.2-90b":
        response = ollama.chat(
            model="llama3.2-vision:90b",
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT},
                {"role": "user", "content": BASE_PROMPT_BEFORE_IMG},
                {"role": "user", "images": [image]},
                {"role": "user", "content": prompt},
            ],
        )
        response = response.message.content
        # remove ale \n in the response
        response = response.replace("\n", "").strip()
    
    elif model == "gemma3:27b":
        response = ollama.chat(
            model="gemma3:27b",
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT},
                {"role": "user", "content": BASE_PROMPT_BEFORE_IMG},
                {"role": "user", "images": [image]},
                {"role": "user", "content": prompt},
            ],
        )
        response = response.message.content
        # remove ale \n in the response
        response = response.replace("\n", "").strip()
    else:
        raise ValueError(f"Invalid model name. Choose either 'openai' or 'llava-7b'. got {model}")

    return response


def parse_json(json_string: str) -> dict:
    """
    Parse a JSON string and return a dictionary.

    Args:
        json_string (str): The JSON string to parse.

    Returns:
        dict: The parsed JSON as a dictionary.
    """
    # Remove any unwanted characters or formatting
    cleaned_json_string = re.sub(r"```json|```", "", json_string).strip()
    parsed_dict = json.loads(cleaned_json_string)
    # remove __ from the caption
    parsed_dict["caption"] = parsed_dict["caption"].replace("_", "")
    # remove the last . if present
    parsed_dict["caption"] = re.sub(r"\.$", "", parsed_dict["caption"])
    # strip the caption
    parsed_dict["caption"] = parsed_dict["caption"].strip()
    
    return parsed_dict

def compute_logits(model: HookedModel, image, generated_json):
    all_logits_with_img = []
    all_logits_without_img = []
    processor = model.get_processor()
    text_tokenizer = model.get_text_tokenizer()
    for item in [generated_json]:
        # Use the new key "caption" instead of "Sentence"
        caption = item["caption"]
        # Remove trailing " ---" or " ---." if present
        caption = re.sub(r" ---\.?$", "", caption)
        # Remove blank tokens such as ___ or __ at the end
        caption = re.sub(r"___\.*$", "", caption)
        caption = re.sub(r"__\.*$", "", caption)
        # Strip any trailing whitespace
        caption = caption.strip()
        item["caption"] = caption

        # Compute logits with image context
        model.use_full_model()
        prompt = f"<image>\n. {caption}"
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        logits = model.predict(inputs=inputs, k=10)
        all_logits_with_img.append(logits)

        # Compute logits without image context
        model.use_language_model_only()
        prompt = f"{caption}"
        inputs = text_tokenizer(text=prompt, return_tensors="pt")
        logits = model.predict(inputs=inputs, k=10)
        all_logits_without_img.append(logits)

    for i, item in enumerate([generated_json]):
        item["logits_without_img"] = all_logits_without_img[i]
        item["logits_with_img"] = all_logits_with_img[i]

    return generated_json


def generate_sentence(
    image,
    selected_caption: str,
    crowd_captions: list[str],
    designer_explanation: str,
    crowd_explanations: list[str],
    model,
    generator_model: Literal["openai", "llava-7b", "llava-34b"],
) -> dict:
    # Create itemized versions of the crowd captions and explanations
    crowd_captions_itemized = "\n".join(
        [f"{i}. {caption}" for i, caption in enumerate(crowd_captions, 1)]
    )
    crowd_explanation_itemized = "\n".join(
        [f"{i}. {explanation}" for i, explanation in enumerate(crowd_explanations, 1)]
    )
    # Fill in the prompt with the context information
    prompt = BASE_PROMPT_AFTER_IMG.format(
        selected_caption=selected_caption,
        crowd_captions_itemized=crowd_captions_itemized,
        designer_explanation=designer_explanation,
        crowd_explanation_itemized=crowd_explanation_itemized,
    )

    # Prepare the image as a base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # Use "PNG" if preferred
    image_bytes = buffered.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # First call: generate candidate sentences with context and image
    # response = CLIENT.chat.completions.create(
    #     model="gpt-4o-2024-08-06",
    #     # model="gpt-4o-mini-2024-07-18",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": [
    #             {"type": "text", "text": BASE_PROMPT_BEFORE_IMG},
    #             {"type": "image_url", "image_url": {
    #                 "url": f"data:image/jpeg;base64,{base64_image}"
    #             }},
    #             {"type": "text", "text": prompt}
    #         ]}
    #     ]
    # )
    response = get_model_response(prompt, base64_image, generator_model)

    generated_json = parse_json(response)
    

    # Compute logits for each candidate caption
    generated_json = compute_logits(model, image, generated_json)

    return {"generated_sentences": generated_json}


def main(
    model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
    n_samples: int = 500,
    generator_model: Literal["openai", "llava-7b", "llava-34b", "llama-3.2-90b", "gemma3:27b"] = "llava-7b",
):
    if generator_model != "openai":
        start_ollama(generator_model)
    model = HookedModel.from_pretrained(model_name, device_map="balanced")
    dataset = load_dataset("nlphuji/whoops", split=f"test[:{n_samples}]")
    print("Model and dataset loaded")

    # Compute the output filename once at the beginning using the start time.
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    hours = now.strftime("%H-%M-%S")
    output_filename = f"./data/openai/visual_counterfactual_{date}_{hours}.json"

    # Create the output directory if it does not exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    # Randomly shuffle the dataset
    dataset = dataset.shuffle(seed=43)
    generated_data = []

    for item in tqdm(dataset, total=len(dataset)):
        try:
            generated_json = generate_sentence(
                item["image"],  # type: ignore
                item["selected_caption"],  # type: ignore
                json.loads(item["crowd_captions"]),  # type: ignore
                item["designer_explanation"],  # type: ignore
                json.loads(item["crowd_explanations"]),  # type: ignore
                model,
                generator_model,
            )
            # Add additional metadata to the generated data
            generated_json["image_id"] = item["image_id"]
            generated_json["selected_caption"] = item["selected_caption"]
            generated_data.append(generated_json)
        except Exception as e:
            print(f"----> Item skipped. Error: {e}")
            continue

        # Save the progress after each iteration
        with open(output_filename, "w") as f:
            json.dump(generated_data, f, indent=4)

    print("Data saved to", output_filename)



if __name__ == "__main__":
    main(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        n_samples=500,
        generator_model="openai"
    )
