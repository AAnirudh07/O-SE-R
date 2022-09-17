# O-SE-R
This repository contains the implementation details for the OCR tool built to convert UML sketches to a code skeleton using Image Processing.

### How to use O-SE-R

#### Prerequisites
- opencv: 4.6.0.66
- numpy: 1.23.2
- Download the custom YOLO weight file here: https://drive.google.com/drive/folders/1Bp7WFeSSX3HlyC1KIU4BPRpBHqo5AbLq?usp=sharing

#### Run O-SE-R
- Detect the UML classes and inheritance symbols: `yolo_img.py`
- Generate code skeleton: `gen_code_skel.py`




## Test Set Results:


      Total Number of Classes: 274
      Number of Classes Identified: 233
      % of Classes Identified: 85.03%

      Total Number of Inheritance Symbols: 50
      Number of Inheritance Symbols Identified: 44
      % of Classes Identified: 88.00%

      Total Number of Associations: 144
      Number of Associations Identified: 64
      % of Associations Identified: 44.44%

      Total Number of Generalizations: 50
      Number of Generalizations Identified: 18
      % of Generalizations Identified: 36%

## Dataset Link: https://www.kaggle.com/datasets/leticiapiucco/handwritten-uml-class-diagrams


