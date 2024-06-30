
#  Runs prompts for tasks Florence-2 model is capable of. See https://huggingface.co/microsoft/Florence-2-large.
#  Requires CUDA Toolkit and some dependencies to run the PyTorch model.
# 
#    1. Install CUDA Toolkit even if you don't use an NVIDIA GPU.
#
#           https://developer.nvidia.com/cuda-downloads (2.9 GB)
# 	
#    2. Install the following dependencies with pip or pip3:
#
#           sudo pip3 install Image transformers torch transformers datasets einops timm flash_attn
#
#    3. Clone the Florence-2-large. You'll need Git-LFS (https://git-lfs.github.com) to clone the binary in full.
#
#           cd florence-2
#           git lfs clone https://huggingface.co/microsoft/Florence-2-large
#
#    4. Run the script.
#
#           python3 1-tasks.py
#
#
#  Windows users installing FlashAttention-2 (flash_attn) may encounter the following errors. I suggest you save
#  yourself some disappointment https://github.com/Dao-AILab/flash-attention/issues/509 and simply run in WSL2. :-)
#
#    - CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
#    - UserWarning: flash_attn was requested, but nvcc was not found.  Are you sure your environment has nvcc available?

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

print("Loading model florence-2-large...")
model = AutoModelForCausalLM.from_pretrained("Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Florence-2-large", trust_remote_code=True)
can_image = Image.open("images/cans/can.jpg")
car_image = Image.open("images/cars/car.jpg")
cat_image = Image.open("images/cats/cat.png").convert("RGB") # Unable to infer channel dimension format? Convert to RGB.

def run_vision_task_on(image: Image, task_prompt: str, text_input=None) -> type[str|dict]:
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

# Caption
print("\n\nTask 1: Caption\n")
prompt = "<CAPTION>"
answer = run_vision_task_on(car_image, prompt)
print(answer)

# Detailed caption
print("\n\nTask 2: Detailed caption\n")
prompt = "<DETAILED_CAPTION>"
answer = run_vision_task_on(car_image, prompt)
print(answer)

# More detailed caption
print("\n\nTask 3: More detailed caption\n")
prompt = "<MORE_DETAILED_CAPTION>"
answer = run_vision_task_on(car_image, prompt)
print(answer)

# Caption to phrase grounding
print("\n\nTask 4: Caption to phrase grounding\n")
prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
answer = results = run_vision_task_on(car_image, prompt, text_input="A green car parked in front of a yellow building.")
print(answer)

# Object detection
print("\n\nTask 5: Object detection\n")
prompt = "<OD>"
answer = run_vision_task_on(car_image, prompt)
print(answer)

# Dense region captioning
print("\n\nTask 6: Dense region captioning\n")
prompt = "<DENSE_REGION_CAPTION>"
answer = run_vision_task_on(can_image, prompt)
print(answer)

# Region proposal
print("\n\nTask 7: Region proposal\n")
prompt = "<REGION_PROPOSAL>"
answer = run_vision_task_on(can_image, prompt)
print(answer)

# OCR
print("\n\nTask 8: OCR\n")
prompt = "<OCR>"
answer = run_vision_task_on(cat_image, prompt)
print(answer)

# OCR with region
print("\n\nTask 9: OCR with region\n")
prompt = "<OCR_WITH_REGION>"
answer = run_vision_task_on(cat_image, prompt)
print(answer)

print("\nDone.")

# Loading model florence-2-large...
#
#
# Task 1: Caption
#
# {'<CAPTION>': '\nA green car parked in front of a yellow building.\n'}
#
#
# Task 2: Detailed caption
#
# {'<DETAILED_CAPTION>': '\nThe image shows a blue Volkswagen Beetle parked in front of a yellow building with two brown doors, surrounded by trees and a clear blue sky.\n'}
#
#
# Task 3: More detailed caption
#
# {'<MORE_DETAILED_CAPTION>': '\nThe image shows a vintage Volkswagen Beetle car parked on a cobblestone street in front of a yellow building with two wooden doors. The car is painted in a bright turquoise color and has a white stripe running along the side. It has two doors on either side of the car, one on top of the other, and a small window on the front. The building appears to be old and dilapidated, with peeling paint and crumbling walls. The sky is blue and there are trees in the background.\n'}
#
#
# Task 4: Caption to phrase grounding
#
# {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[34.880001068115234, 158.63999938964844, 583.3599853515625, 374.6399841308594], [0.3199999928474426, 4.079999923706055, 639.0399780273438, 305.03997802734375]], 'labels': ['A green car', 'a yellow building']}}
#
#
# Task 5: Object detection
#
# {'<OD>': {'bboxes': [[34.23999786376953, 160.0800018310547, 597.4400024414062, 371.7599792480469], [272.32000732421875, 242.1599884033203, 303.67999267578125, 246.95999145507812], [454.0799865722656, 276.7200012207031, 553.9199829101562, 370.79998779296875], [96.31999969482422, 280.55999755859375, 198.0800018310547, 371.2799987792969]], 'labels': ['car', 'door handle', 'wheel', 'wheel']}}
#
#
# Task 6: Dense region captioning
#
# {'<DENSE_REGION_CAPTION>': {'bboxes': [[43.5, 1.1355000734329224, 355.5, 722.5565185546875], [370.5, 104.0875015258789, 680.5, 720.2855224609375], [692.5, 104.0875015258789, 998.5, 720.2855224609375]], 'labels': ['Coca-Cola Premium Quality 330ml can', 'Coca Cola can 12 fl oz 355 ml', 'Coca Cola can 12 fl oz 355 ml']}}
#
#
# Task 7: Region proposal
#
# {'<REGION_PROPOSAL>': {'bboxes': [[43.5, 1.1355000734329224, 355.5, 722.5565185546875], [369.5, 104.0875015258789, 680.5, 720.2855224609375], [691.5, 104.0875015258789, 998.5, 720.2855224609375]], 'labels': ['', '', '']}}
#
#
# Task 8: OCR
#
# {'<OCR>': '\nI CAN HASCHEEZBURGER?ICANHASCHEEZEURGER.COM\n'}
#
#
# Task 9: OCR with region
#
# {'<OCR_WITH_REGION>': {'quad_boxes': [[62.653499603271484, 10.54699993133545, 196.96949768066406, 10.54699993133545, 196.96949768066406, 38.80500030517578, 62.653499603271484, 38.80500030517578], [28.528499603271484, 48.755001068115234, 234.09750366210938, 48.755001068115234, 234.09750366210938, 78.6050033569336, 28.528499603271484, 78.6050033569336], [2.5935001373291016, 386.6570129394531, 163.3905029296875, 386.6570129394531, 163.3905029296875, 396.60699462890625, 2.5935001373291016, 396.60699462890625]], 'labels': ['</s>I CAN HAS', 'CHEEZBURGER?', 'ICANHASCHEEZEURGER.COM C']}}
#
# Done.
