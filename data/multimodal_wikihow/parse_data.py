import pickle
import json

# list of dict with keys: dict_keys(['file_id', 'goal', 'goal_description', 'category_hierarchy', 'methods'])
data = [json.loads(row.strip()) for row in open("WikihowText_data.json")]

data_dict = dict()

for row in data:
    file_id = row["file_id"]
    goal = row["goal"]
    goal_description = row["goal_description"]
    for method in row["methods"]:
        method_name = method["name"]
        for step in method["steps"]:
            data_dict[step["step_id"]] = {
                **step,
                "file_id": file_id,
                "goal": goal,
                "goal_description": goal_description,
                "method": method_name
            }

json.dump(data_dict, open("wikihow_text_by_step_id.json","w"))