# Instruction
This is the instruction how to create the Multimodal WikiHow Summarization data

This assumes you have downloaded `wiki_images.zip` and `WikihowText_data.json` from [WikiHow VGSI](https://github.com/YueYANG1996/wikiHow-VGSI).

First run `python parse_data.py` to generate a mapping of step to the necessary information.

Then, run `python create_data.py`.

# Requirements
- python 3
- datasets
- sklearn

# Output
A DatasetDict instance consisting of `train, validation, test` Dataset objects. Each Dataset consists of:
- document(string)
- summary(string)
- images(string): the image id corresponding to the step_id

# License
Content on wikiHow can be shared under a [Creative Commons license](http://creativecommons.org/licenses/by-nc-sa/3.0/).