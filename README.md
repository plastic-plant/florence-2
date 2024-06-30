Last week, Microsoft's Azure AI team dropped **Florence-2** on Hugging Face. Let's play.

The model can handle a variety of vision tasks using a prompt-based representation. Should excel at tasks such as captioning, object detection, visual grounding and segmentation, performing on par or better than many large vision models out there. Comes in two sizes, with [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) as light as 1.54 GB finetuned for OCR.

### python 1-tasks.py

Runs prompts for tasks Florence-2 model is capable of, requires CUDA Toolkit.

1. Install CUDA Toolkit even if you don't use an NVIDIA GPU.
```
    https://developer.nvidia.com/cuda-downloads (2.9 GB)
``` 	
2. Install the following dependencies with pip or pip3:
```bash
    sudo pip3 install Image transformers torch transformers datasets einops timm flash_attn
```
3. Clone the Florence-2-large. You'll need Git-LFS (https://git-lfs.github.com) to clone the binary in full.
```bash
        cd florence-2
        git lfs clone https://huggingface.co/microsoft/Florence-2-large
```
4. Run the script.
```bash
    python3 1-tasks.py
```

Windows users installing FlashAttention-2 (flash_attn) may encounter the following errors. I suggest you save 
yourself [some disappointment](https://github.com/Dao-AILab/flash-attention/issues/509) and simply run in WSL2. :-)

- _CUDA_HOME environment variable is not set. Please set it to your CUDA install root._
- __UserWarning: flash_attn was requested, but nvcc was not found.  Are you sure your environment has nvcc available?_

#### Output
```json
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

