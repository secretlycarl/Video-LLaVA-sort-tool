import torch
import os
import shutil
import traceback
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def process_video(video_path, question, model_path, cache_dir, device, load_4bit, load_8bit, model_name, tokenizer, model, processor, conv_mode, conv, roles):
    disable_torch_init()
    video = video_path
    inp = question

    video_processor = processor['video']

    video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)

    # Remove the </s> at the end of the output
    outputs = outputs.replace('</s>', '')

    return outputs

def main():
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    conv_mode = "llava_v1"

    while True:
        folder_path = input("Enter the folder path (or 'exit' to quit): ")
        if folder_path.lower() == 'exit':
            break

        question = input("Enter the question: ")

        video_answers = {}

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".mp4", ".mov")):
                video_path = os.path.join(folder_path, filename)
            
                # Create a new instance of the conversation template for each video
                conv = conv_templates[conv_mode].copy()
                roles = conv.roles

                print(f"Processing video: {filename}")
            
                try:
                    answer = process_video(video_path, question, model_path, cache_dir, device, load_4bit, load_8bit, model_name, tokenizer, model, processor, conv_mode, conv, roles)

                    # Store the video-answer pair
                    video_answers[video_path] = answer
                except Exception as e:
                   print(f"Error processing video {video_path}: {e}")
                   print(traceback.format_exc())
                   continue
        
        # Sort the videos based on the model's answer
        for video_path, answer in video_answers.items():
            destination_folder = os.path.join(folder_path, answer)
            os.makedirs(destination_folder, exist_ok=True)
            shutil.move(video_path, destination_folder)

if __name__ == '__main__':
    main()