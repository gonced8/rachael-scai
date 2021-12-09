import json

path = "../../dataset/qrecc-test.json"
results = []
with open(path) as f:
    data = json.load(f)
    for sample in data:
        answer = sample["Question"]
        result = {
            "Model_answer": answer,
            "Conversation_no": sample["Conversation_no"],
            "Turn_no": sample["Turn_no"],
        }
        results.append(result)
print(len(results), "samples")

run_path = "basic.json"

with open(run_path, "w") as output_file:
    json.dump(results, output_file, indent=4)
