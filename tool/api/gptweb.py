import openai
import base64
import os
from flask import Flask, request, jsonify
import argparse
import time

app = Flask(__name__)

API_KEY = 'your api key'
openai.api_key = API_KEY


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
def test_gpt4o_with_local_images(api_key, prompt, image_paths, model="gpt-4o", max_tokens=300, max_retries=6, retry_delay=3):
    openai.api_key = api_key

    base64_images = [encode_image_to_base64(image_path) for image_path in image_paths]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ] + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                } for base64_image in base64_images
            ]
        }
    ]
    
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            retries += 1
            print(f"Error during GPT-4o request: {e}")
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(retry_delay) 
            else:
                print("Max retries exceeded.")
                return None

@app.route('/test_gpt4o', methods=['POST'])
def test_gpt4o():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({'error': 'Prompt not provided'}), 400

        image_files = data.get('images', [])
        image_paths = []
        if image_files:
            for idx, image_data in enumerate(image_files):
                if image_data.startswith("data:image"): 
                    image_content = base64.b64decode(image_data.split(',')[1])
                else:
                    image_content = base64.b64decode(image_data)
                
                image_path = f"temp_image_{idx}.png"
                with open(image_path, "wb") as img_file:
                    img_file.write(image_content)
                image_paths.append(image_path)

        result = test_gpt4o_with_local_images(API_KEY, prompt, image_paths)
        for image_path in image_paths:
            os.remove(image_path)

        if result:
            return jsonify({'response': result}), 200
        else:
            return jsonify({'error': 'No response from GPT-4o'}), 500

    except Exception as e:
        return jsonify({'error': f"Error occurred: {str(e)}"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)