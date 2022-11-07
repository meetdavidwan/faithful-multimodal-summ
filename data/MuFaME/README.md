the file `MuFaME` consists of the human annotations in the 2 csv files corresponding to the validation and test file.

Each file consists of the following columns:
- doc_id (str): step id
- model (str): model from the list `reference, pegasus, t5, vlbart, mof`
- document (str)
- summary (str)
- image (str): image url. In case it expired, please use the step id and get the corresponding image from [WikiHow VGSI](https://github.com/YueYANG1996/wikiHow-VGSI).
- doc_factuality (int): 1 if the worker labeled factual with respect to document
- img_factuality (int): 1 if the worker labeled factual with respect to image
- comb_factuality (float): the combined judgment (average across the two factuality judgment)
- wid (int): 0,1,2 indicating which worker