import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def read_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {filename}.")
        return None

def extract_phq8_scores(text):
    total_score = 0
    remaining_text = text
    
    for _ in range(8):
        score_index = remaining_text.find('Score:')
        if score_index == -1:
            break
        
        score_str = remaining_text[score_index + 7:score_index + 10]
        try:
            score = int(float(score_str))
            total_score += score
        except ValueError:
            break
        
        remaining_text = remaining_text[score_index + 10:]
    
    return total_score

def scores_to_binary(scores, threshold=10):
    return [1 if score >= threshold else 0 for score in scores]

def load_ground_truth():
    csv_path = 'Llama3-8B-LoRa/full_test_split.csv'
    df = pd.read_csv(csv_path)
    scores = df['PHQ_Score'].tolist()
    del scores[45]  # Remove specific participant
    return scores_to_binary(scores)

def load_predictions(model_name):
    with open(f'{model_name}.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    
    participant_responses = content.split("Participant_ID:")[1:]
    predicted_scores = [extract_phq8_scores(response) for response in participant_responses]
    return scores_to_binary(predicted_scores)

def compute_statistics(results):
    dict_metrics = [m for m in results if isinstance(m, dict)]
    scalars = [m for m in results if isinstance(m, (int, float))]
    
    # Calculate averages for dictionary metrics
    average_metrics = {k: np.mean([d[k] for d in dict_metrics]) for k in dict_metrics[0]}
    std_metrics = {k: np.std([d[k] for d in dict_metrics]) for k in dict_metrics[0]}
    
    # Calculate average and standard deviation of scalars
    average_scalar = np.mean(scalars)
    std_scalar = np.std(scalars)
    
    return average_metrics, std_metrics, average_scalar, std_scalar

def main():
    seeds = [40, 41, 42, 43, 44]
    class_labels = {'0': 'Non depressed', '1': 'Depressed'}
    
    results = []
    class_f1_scores = {label: 0 for label in class_labels.keys()}
    
    # Load ground truth
    ground_truth = load_ground_truth()
    
    # Process each seed
    for seed in seeds:
        model_name = f'data.json_phq8_d_test.json_{seed}'
        
        # Load predictions
        predictions = load_predictions(model_name)
        
        # Generate classification report
        report = classification_report(
            ground_truth, predictions, 
            digits=3, output_dict=True, zero_division=0
        )
        
        print(f"Results for seed {seed}:")
        print(report)
        
        # Accumulate class-wise F1 scores
        for class_key in class_f1_scores.keys():
            if class_key in report:
                class_f1_scores[class_key] += report[class_key]['f1-score']
        
        # Store macro average and accuracy
        results.append(report['macro avg'])
        results.append(report['accuracy'])
    
    # Compute and display statistics
    avg_metrics, std_metrics, avg_scalar, std_scalar = compute_statistics(results)
    
    print("\n" + "="*50)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*50)
    
    print("\nAveraged Metrics:")
    print(f"- Precision: {avg_metrics['precision']:.3f} (std: {std_metrics['precision']:.3f})")
    print(f"- Recall: {avg_metrics['recall']:.3f} (std: {std_metrics['recall']:.3f})")
    print(f"- F1-Score: {avg_metrics['f1-score']:.3f} (std: {std_metrics['f1-score']:.3f})")
    print(f"- Accuracy: {avg_scalar:.3f} (std: {std_scalar:.3f})")
    
    print("\nClass-wise Average F1-Scores:")
    for class_key, class_name in class_labels.items():
        avg_f1 = class_f1_scores[class_key] / len(seeds)
        print(f"- {class_name}: {avg_f1:.3f}")

if __name__ == "__main__":
    main()
