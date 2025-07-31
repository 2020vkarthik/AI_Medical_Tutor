# pip install rouge-score nltk bert-score sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score

def evaluate_mcq(model_answer, gold_answer):
    return model_answer.strip().lower() == gold_answer.strip().lower()

def evaluate_explanation(model_explanation, gold_explanation):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(model_explanation, gold_explanation)['rougeL'].fmeasure
    bleu_score = sentence_bleu([gold_explanation.split()], model_explanation.split())
    P, R, F1 = score([model_explanation], [gold_explanation], lang="en")
    return {
        "ROUGE-L": rouge_score,
        "BLEU": bleu_score,
        "BERTScore": F1.tolist()[0]
    }

def evaluate_case_based(model_answer, gold_answer):
    P, R, F1 = score([model_answer], [gold_answer], lang="en")
    return {"BERTScore": F1.tolist()[0]}

def evaluate_short_answer(model_answer, gold_answer):
    return evaluate_explanation(model_answer, gold_answer)

# Gold Standard Responses
gold_standard = {
    "MCQ1": {"answer": "D", "explanation": "Dehydration reduces fluid levels in the body, leading to decreased urine output as the kidneys try to conserve water."},
    "MCQ2": {"answer": "B", "explanation": "Diarrhea leads to excessive water and electrolyte loss through stool, causing dehydration."},
    "Case1": "Dehydration due to decreased fluid intake or increased fluid loss.",
    "Case2": "Administer oral rehydration therapy (ORT) with a solution containing electrolytes and carbohydrates.",
    "Short1": "Dehydration disrupts electrolyte balance by increasing sodium concentration in the blood, which can lead to cellular dysfunction and organ impairment.",
    "Short2": "Lack of vasopressin (ADH), which causes excessive water loss through increased urine production."
}

# Model Responses
model_responses = {
    "Model1": {
        "MCQ1": {"answer": "D", "explanation": "Dehydration can lead to decreased urine output due to insufficient fluid intake."},
        "Case1": "Dehydration due to decreased fluid intake or increased fluid loss.",
        "Short1": "Dehydration can disrupt the body's electrolyte balance by causing an imbalance in the concentration of water and electrolytes in the body, leading to changes in the function of cells and organs."
    },
    "Model2": {
        "MCQ2": {"answer": "B", "explanation": "In patients with diarrhea, the gut lining is damaged, leading to increased permeability and loss of electrolytes and water into the stool. As a result, the body tries to compensate by increasing urine production to eliminate excess sodium and water, leading to dehydration."},
        "Case2": "Administer oral rehydration therapy (ORT) with a solution containing electrolytes and carbohydrates.",
        "Short2": "Increased urine production due to the absence of vasopressin (ADH), leading to excessive water loss in the urine."
    }
}

# Evaluation
results = {}
for model, responses in model_responses.items():
    results[model] = {}
    for qid, response in responses.items():
        if isinstance(response, dict):  # MCQ type
            correctness = evaluate_mcq(response["answer"], gold_standard[qid]["answer"])
            explanation_scores = evaluate_explanation(response["explanation"], gold_standard[qid]["explanation"])
            results[model][qid] = {"Correct": correctness, "Explanation Scores": explanation_scores}
        else:  # Case-based or short answer
            if "Case" in qid:
                results[model][qid] = evaluate_case_based(response, gold_standard[qid])
            else:
                results[model][qid] = evaluate_short_answer(response, gold_standard[qid])

# Print Results
import json
print(json.dumps(results, indent=4))



