
#  Prompts Florence-2 for <OD> object detection on an image, then draws caption, description labels and bounding boxes.
#
#    sudo pip install bbox-visualizer opencv-python
#    python3 5-identify.py

# Output:
#
#    Loading model.
#    Prompting model for caption and description.
#    Prompting model for image object detection.
#    Found objects ['banana', 'grapes', 'orange', 'pear', 'pear'] with bounding boxes in fruit.jpg.
#    Drawing caption, description, bounding boxes and labels, saving to fruit-with-labels.jpg

import textwrap
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import os
import json
import bbox_visualizer as bbv
import cv2

def prompt(image: Image, task_prompt: str, text_input=None) -> type[str|dict]:
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

print("Loading model.")
model = AutoModelForCausalLM.from_pretrained("Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Florence-2-large", trust_remote_code=True)

print("Prompting model for caption and description.")
image = Image.open("images/fruit/fruit.jpg")
caption = prompt(image, "<CAPTION>").get("<CAPTION>") # A bowl filled with fruit sitting on top of a table.
description = prompt(image, "<MORE_DETAILED_CAPTION>").get("<MORE_DETAILED_CAPTION>") # The image is a photograph of a fruit bowl. The bowl is made of a light-colored material and has a smooth texture. It is filled with a variety of fresh fruits, including a pear, an orange, a cantaloupe, a banana, and a bunch of green grapes. The fruits are arranged in a way that they are overlapping each other, creating a colorful and appetizing display. The background is a plain beige color, making the bowl the focal point of the image.

print("Prompting model for image object detection.")
image = Image.open("images/fruit/fruit.jpg")
answer = prompt(image, "<OD>") # {'<OD>': {'bboxes': [[733.125, 67.5, 1015.625, 520.5], [561.875, 345.5, 990.625, 844.5], [334.375, 371.5, 629.375, 603.5], [169.375, 150.5, 464.375, 482.5]], 'labels': ['banana', 'grapes', 'orange', 'pear']}}
object = answer.get("<OD>")
labels = object.get("labels") # ['banana', 'grapes', 'orange', 'pear']
bboxes = object.get("bboxes") # [[733.125, 67.5, 1015.625, 520.5], .., .., ..]
bboxes_int = [[int(f) for f in bbox] for bbox in bboxes] # [[733, 67, 1015, 520], .., .., ..]
print(f"Found objects {labels} with bounding boxes in fruit.jpg.")


print("Drawing caption, description, bounding boxes and labels, saving to fruit-with-labels.jpg")
cvimage = cv2.imread("images/fruit/fruit.jpg")

# Caption
cv2.putText(cvimage, caption.strip(), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)

# Description
y = 180
for line in textwrap.wrap(description, width=80):
    cv2.putText(cvimage, line.strip(), (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    y += 30

# Boundimg boxes
pear_indexes = [i for i, label in enumerate(labels) if label == 'pear']
pear_bboxes = [bboxes_int[i] for i in pear_indexes]
for pear in pear_bboxes:
    cv2.rectangle(cvimage, (pear[0], pear[1]), (pear[2], pear[3]), (255, 0, 0), 3, cv2.LINE_AA)

# Labels    
im_array = bbv.draw_multiple_flags_with_labels(cvimage, labels, bboxes_int)
im = Image.fromarray(im_array)
im.save("images/fruit/fruit-with-labels.jpg")
