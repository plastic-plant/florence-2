Last week, Microsoft's Azure AI team dropped **Florence-2** on Hugging Face. Let's play.

The model can handle a variety of vision tasks using a prompt-based representation. Should excel at tasks such as captioning, object detection, visual grounding and segmentation, performing on par or better than many large vision models out there. Comes in two sizes, with [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) as light as 1.54 GB finetuned for OCR.


## Vision tasks with Florence-2

Runs prompts for tasks Florence-2 model is capable of. Requires Git-LFS and CUDA Toolkit.

<details><summary>See step-by-step instructions</summary>

1. Install CUDA Toolkit (even if you don't use an NVIDIA GPU).
```
   https://developer.nvidia.com/cuda-downloads (2.9 GB)
``` 	
2. Install the following dependencies with pip:
```bash
   sudo pip3 install Image transformers torch transformers datasets einops timm flash_attn
```
3. Clone the Florence-2-large. You'll need Git-LFS (https://git-lfs.github.com) to clone the binary in full.
```bash
   cd florence-2
   git lfs clone https://huggingface.co/microsoft/Florence-2-large
```
4. Run the script.
</details>

```bash
   python 1-tasks.py
```

<details><summary>Notes for Windows users</summary>

Windows users installing FlashAttention-2 (package `flash_attn`) may encounter the following errors. I suggest you save yourself [some disappointment](https://github.com/Dao-AILab/flash-attention/issues/509) and simply run in WSL2. :-)
- CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
- UserWarning: flash_attn was requested, but nvcc was not found. Are you sure your environment has nvcc available?

In Windows, you're used to `python` and `pip`. In most Linux distro, it's `python3` and `pip3`.

</details>

<details><summary>See example ouput</summary>

#### Output
```
Loading model florence-2-large...


Task 1: Caption

{'<CAPTION>': '\nA green car parked in front of a yellow building.\n'}


Task 2: Detailed caption

{'<DETAILED_CAPTION>': '\nThe image shows a blue Volkswagen Beetle parked in front of a yellow building with two brown doors, surrounded by trees and a clear blue sky.\n'}


Task 3: More detailed caption

{'<MORE_DETAILED_CAPTION>': '\nThe image shows a vintage Volkswagen Beetle car parked on a cobblestone street in front of a yellow building with two wooden doors. The car is painted in a bright turquoise color and has a white stripe running along the side. It has two doors on either side of the car, one on top of the other, and a small window on the front. The building appears to be old and dilapidated, with peeling paint and crumbling walls. The sky is blue and there are trees in the background.\n'}


Task 4: Caption to phrase grounding

{'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[34.880001068115234, 158.63999938964844, 583.3599853515625, 374.6399841308594], [0.3199999928474426, 4.079999923706055, 639.0399780273438, 305.03997802734375]], 'labels': ['A green car', 'a yellow building']}}


Task 5: Object detection

{'<OD>': {'bboxes': [[34.23999786376953, 160.0800018310547, 597.4400024414062, 371.7599792480469], [272.32000732421875, 242.1599884033203, 303.67999267578125, 246.95999145507812], [454.0799865722656, 276.7200012207031, 553.9199829101562, 370.79998779296875], [96.31999969482422, 280.55999755859375, 198.0800018310547, 371.2799987792969]], 'labels': ['car', 'door handle', 'wheel', 'wheel']}}


Task 6: Dense region captioning

{'<DENSE_REGION_CAPTION>': {'bboxes': [[43.5, 1.1355000734329224, 355.5, 722.5565185546875], [370.5, 104.0875015258789, 680.5, 720.2855224609375], [692.5, 104.0875015258789, 998.5, 720.2855224609375]], 'labels': ['Coca-Cola Premium Quality 330ml can', 'Coca Cola can 12 fl oz 355 ml', 'Coca Cola can 12 fl oz 355 ml']}}


Task 7: Region proposal

{'<REGION_PROPOSAL>': {'bboxes': [[43.5, 1.1355000734329224, 355.5, 722.5565185546875], [369.5, 104.0875015258789, 680.5, 720.2855224609375], [691.5, 104.0875015258789, 998.5, 720.2855224609375]], 'labels': ['', '', '']}}


Task 8: OCR

{'<OCR>': '\nI CAN HASCHEEZBURGER?ICANHASCHEEZEURGER.COM\n'}


Task 9: OCR with region

{'<OCR_WITH_REGION>': {'quad_boxes': [[62.653499603271484, 10.54699993133545, 196.96949768066406, 10.54699993133545, 196.96949768066406, 38.80500030517578, 62.653499603271484, 38.80500030517578], [28.528499603271484, 48.755001068115234, 234.09750366210938, 48.755001068115234, 234.09750366210938, 78.6050033569336, 28.528499603271484, 78.6050033569336], [2.5935001373291016, 386.6570129394531, 163.3905029296875, 386.6570129394531, 163.3905029296875, 396.60699462890625, 2.5935001373291016, 396.60699462890625]], 'labels': ['</s>I CAN HAS', 'CHEEZBURGER?', 'ICANHASCHEEZEURGER.COM C']}}

Done.
```
</details>


## Optical character recognition

Prompts Florence-2 for `<OCR>` on a [folder of images](images/monarchs/) and writes to [text files](images/texts/) in the same folder.

```bash
   python 2-ocr.py
```

![Example picture with a mix of printed and handwritten text](images/monarchs/Frederick%20II.png?raw=true)

<details><summary>See example ouput</summary>

#### Output
```
Loading model florence-2-large.
OCR on a folder of images:

images/monarchs/Charlemagne.png: Charlemagne, also known as Charles the Great, was King of the Franks and Lombards before becoming thefirst Holy Roman Emperor in 800. His reign is marked by the Carolinian Renaissance, a revival of art,culture, and learning based on classical models, which laid the foundations for medieval Europeancivilization.
images/monarchs/Charlemagne.txt: saved.

images/monarchs/Frederick II.png: Frederick II,KNOWN AS Frederick the Great,was King of Prussia who modernized the Prussian state and made it a majorEuropean power. He is celebrated for his military victories, patronage of thearts and philosophy, and his efforts to improve the legal and administrativesystems of Prussia, which set the stage for its rise in the 19th century.Source: Chal GPT + Patrick
images/monarchs/Frederick II.txt: saved.

images/monarchs/Henry VIII.png: Henry VIII was the King of England who is perhaps best known for his six marriages and his role in theseparation of the Church of England from the Roman Catholic Church. His reign was marked by significantchanges in religion, government, and society, including the dissolution of the monasteries and theestablishment of the Royal Navy.
images/monarchs/Henry VIII.txt: saved.

images/monarchs/Louis XIV.png: Known as the Sun King, Louis XIVreigned as King of France for 72years, the longest recorded of anymonarch in European history. Hisreign saw the expansion of frenchinfluence through a series of wars,the establishment of absolutemonarchy, and the construction ofthe opulent Palace of Versailles,symbolizing his power and grandeur.
images/monarchs/Louis XIV.txt: saved.

images/monarchs/William IV.png: William's House of Honoree in London, England in 1881. He was in 1882 in 1887. He married his children in 1818. In 1881, he married Princess Princess Princess Anne in 1884. The children in London in 1891 and 1891. He became a child in 1883. The child was born in London and became a princess in 1894. The mother of Queen Elizabeth II. The son of King George III. The Queen of England was born of England. The queen of England, the Queen of Scotland was born with her son of Great Britain.William's house of Honour in London. He succeeded his son of George II.The Queen's son of Britain. The King of England is born in England. He is born with his mother of England in England, and the Queen's mother of Scotland. The Prince of Wales was born to be born in Britain. William's son is born from England. His son of England and his mother is born to England. William was born from Britain.
images/monarchs/William IV.txt: saved.
```

</details>


## Retrieval-augmented generation

Stores the contents of text files in a Chroma database together with vectors generated by NoMAD-Attention embeddings function. Overwrites existing chroma.sqlite3 database file in the chroma folder for testing purposes. Requires Ollama and the nomic-embed-text model. See https://ollama.com/library.

```bash
   install https://ollama.com/download  # Runs at localhost:11434
   ollama pull nomic-embed-text         # Size: 275 MB
   ollama pull gemma2                   # Size: 5.5 GB

   pip install chromadb langchain langchain_community getch

   python 3-vectors.py
```

Use the Chroma database to search for content by vector cosin similarity, then feed Gemma a prompt with retrieved information as context to answer the question.

```bash
   python 4-ask.py
```

<details><summary>See example ouput</summary>

#### Output

```
python 3-vectors.py

   Removing existing chroma database folder.
   Storing documents with BERT (nomic-embed-text) embeddings in ./chroma/ folder.
   Documents stored.
```

File `./chroma/chroma.sqlite3` is created.

```
python 4-ask.py

   Answering questions on documents matching in Chroma database with GEMMA-2.
   Ask a question, press [CTRL+C] or [ENTER] without a question to quit.

   Ask: Can you give me a summary?

   Answer: The provided documents offer biographies of several European monarchs: Louis XIV of France, William IV of the United Kingdom, Henry VIII of England, Frederick II of Prussia, and Charlemagne.  Each document highlights their significant achievements, reign lengths, and lasting impacts on their respective countries and Europe.
```
```
   Ask: Please list them by name and birth year in the order of birth.

   Answer: Here is the list of monarchs by name and birth year, ordered from earliest to latest:

   1. **Charlemagne** (c. 742)
   2. **Louis XIV** (1638)
   3. **Frederick II** (1712)
   4. **William IV** (1765)
   5. **Henry VIII** (1491)
```
```
   Ask: Who established the Royal Navy?

   Answer: The context states that Henry VIII established the Royal Navy.
```
```
   Ask: Can you list well known aliases for monarchs?

   Answer: Here are some well-known aliases for monarchs from the provided text:

   * **Louis XIV:** The Sun King
   * **William IV:** The Sailor King
   * **Frederick II:** Frederick the Great
   * **Henry VIII:**  (While not an alias, he is often referred to as "Henry VIII")
   * **Charlemagne:** Charles the Great


   Let me know if you have any other questions!
```

</details>


## Labels, boxes and descriptions

Prompts Florence-2 for object detection on an image, then draws caption, description labels and bounding boxes with cv2 and bbv.

```bash
   pip install bbox-visualizer opencv-python
   python 5-identify.py
```

Note how the cantaloupe melon is identified in `<MORE_DETAILED_CAPTION>` prompt, but not labelled by `<OD>` prompt. Also see how well the partial pear is recognised.

![Example output image of fruit with caption, description, bounding boxes and labels](images/fruit/fruit-with-labels.jpg?raw=true)

#### Output
```
   Loading model.
   Prompting model for caption and description.
   Prompting model for image object detection.
   Found objects ['banana', 'grapes', 'orange', 'pear', 'pear'] with bounding boxes in fruit.jpg.
   Drawing caption, description, bounding boxes and labels, saving to fruit-with-labels.jpg

```

</details>


## ONNX version

Xenova uploaded a ONXX conversion of the torch model on the onnx-community repo. See [example code](https://huggingface.co/onnx-community/Florence-2-large), try out [Florence-2 demo with WebGPU](https://huggingface.co/spaces/Xenova/florence2-webgpu).

```bash
   cd florence-2
   git lfs clone https://huggingface.co/microsoft/Florence-2-large
```

Use the ONNX model with [ML.net tutorial](https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/object-detection-onnx).

