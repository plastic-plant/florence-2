
#  Prompts Florence-2 for <OCR> on a folder of images and write the text in images to text files in the folder.
# 
#    python3 2-ocr.py

# Output:
#
#   Loading model florence-2-large.
#   OCR on a folder of images:
#
#   images/monarchs/Charlemagne.png: Charlemagne, also known as Charles the Great, was King of the Franks and Lombards before becoming thefirst Holy Roman Emperor in 800. His reign is marked by the Carolinian Renaissance, a revival of art,culture, and learning based on classical models, which laid the foundations for medieval Europeancivilization.
#   images/monarchs/Charlemagne.txt: saved.
#
#   images/monarchs/Frederick II.png: Frederick II,KNOWN AS Frederick the Great,was King of Prussia who modernized the Prussian state and made it a majorEuropean power. He is celebrated for his military victories, patronage of thearts and philosophy, and his efforts to improve the legal and administrativesystems of Prussia, which set the stage for its rise in the 19th century.Source: Chal GPT + Patrick
#   images/monarchs/Frederick II.txt: saved.
#
#   images/monarchs/Henry VIII.png: Henry VIII was the King of England who is perhaps best known for his six marriages and his role in theseparation of the Church of England from the Roman Catholic Church. His reign was marked by significantchanges in religion, government, and society, including the dissolution of the monasteries and theestablishment of the Royal Navy.
#   images/monarchs/Henry VIII.txt: saved.
#
#   images/monarchs/Louis XIV.png: Known as the Sun King, Louis XIVreigned as King of France for 72years, the longest recorded of anymonarch in European history. Hisreign saw the expansion of frenchinfluence through a series of wars,the establishment of absolutemonarchy, and the construction ofthe opulent Palace of Versailles,symbolizing his power and grandeur.
#   images/monarchs/Louis XIV.txt: saved.
#
#   images/monarchs/William IV.png: William's House of Honoree in London, England in 1881. He was in 1882 in 1887. He married his children in 1818. In 1881, he married Princess Princess Princess Anne in 1884. The children in London in 1891 and 1891. He became a child in 1883. The child was born in London and became a princess in 1894. The mother of Queen Elizabeth II. The son of King George III. The Queen of England was born of England. The queen of England, the Queen of Scotland was born with her son of Great Britain.William's house of Honour in London. He succeeded his son of George II.The Queen's son of Britain. The King of England is born in England. He is born with his mother of England in England, and the Queen's mother of Scotland. The Prince of Wales was born to be born in Britain. William's son is born from England. His son of England and his mother is born to England. William was born from Britain.
#   images/monarchs/William IV.txt: saved.

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import os
import json

def get_image_paths() -> list:
    image_paths = []
    for image_file in os.listdir("images/monarchs"): # or glob
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            image_path = os.path.join("images/monarchs", image_file)
            image_paths.append(image_path)
    return image_paths

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

print("Loading model florence-2-large.")
model = AutoModelForCausalLM.from_pretrained("Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Florence-2-large", trust_remote_code=True)

print("OCR on a folder of images:")
for image_path in get_image_paths():
    save_path = image_path.replace(".png", ".txt")
    image = Image.open(image_path).convert("RGB") # Unable to infer channel dimension format? Convert to RGB.
    answer = prompt(image, "<OCR>") # {'<OCR>': '\nI CAN HASCHEEZBURGER?ICANHASCHEEZEURGER.COM\n'}
    
    if isinstance(answer, str):
        text = json.loads(answer).get("<OCR>")
    else:
        text = answer.get("<OCR>")
    
    print(f"\n{image_path}: {text}")
    with open(save_path, "w") as file:
        file.write(text)
        print(f"{save_path}: saved.")
            
