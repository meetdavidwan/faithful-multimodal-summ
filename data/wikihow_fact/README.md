# Instruction
This is the instruction how to create the WikiHowFact

This assumes you have downloaded `wiki_images.zip` and `WikihowText_data.json` from [WikiHow VGSI](https://github.com/YueYANG1996/wikiHow-VGSI).
Also make sure you have `test_step_random.p, test_step_category.p, test_step_similarity.p`.

First run `python parse_data.py` to generate a mapping of step to the necessary information.

Then, run `python create_data.py`.

# Requirements
- python 3
- datasets

# Output
A DatasetDict instance consisting of `train, validation, test` Dataset objects.

Each Dataset consists of:
- id(string)
- document(string)
- image(string): the image id which is the step_id
- goal(string): the goal of the candidate step
- method(string): the method of the candidate step
- step(string): the summary sentence
- label(int): 1 if it the true summary else 0

# License
Content on wikiHow can be shared under a [Creative Commons license](http://creativecommons.org/licenses/by-nc-sa/3.0/).