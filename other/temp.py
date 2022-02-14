reference = {
    "id": "420",
    "answers": {
        "answer_start": [0],
        "text": [
            "Ryan Dunn's Porsche 911 GT3 veered off the road, struck a tree, and burst into flames in West Goshen Township, Chester County, Pennsylvania."
        ],
    },
}
prediction = {
    "id": "420",
    "prediction_text": "The Florida Department of Law Enforcement concluded that Dunn's death was a homicide caused by a single gunshot wound to the chest.",
    "no_answer_probability": 0.0,
}
score = metric.compute(predictions=[prediction], references=[reference])


score2 = metric2.compute(
    predictions=[
        "The Florida Department of Law Enforcement concluded that Dunn's death was a homicide caused by a single gunshot wound to the chest."
    ],
    references=[
        "Ryan Dunn's Porsche 911 GT3 veered off the road, struck a tree, and burst into flames in West Goshen Township, Chester County, Pennsylvania."
    ],
    use_agregator=False,
)

result = {
    # TODO: THIS IS NOT GOOD...
    "Exact match": score["exact"] / 100,
    "F1": score["f1"] / 100,
    "ROUGEL-F1": score2,
}

print(result)


score2 = metric2.compute(
    predictions=["What were the circumstances of Dunn's death?"],
    references=["What were the circumstances of Ryan Dunn's death?"],
    use_agregator=False,
)
score2["rouge1"]
