import pandas as pd
from sklearn.metrics import classification_report

def load_ground_truth(csv_path, exclude_participants=None):
    df = pd.read_csv(csv_path)
    if exclude_participants:
        df.drop(df[df["Participant_ID"].isin(exclude_participants)].index, inplace=True)
    return df['PHQ8_Score'].tolist()

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

def load_predictions(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    participant_responses = content.split("Participant_ID:")[1:]
    predicted_scores = [extract_phq8_scores(response) for response in participant_responses]
    return scores_to_binary(predicted_scores)

def compute_averaged_metrics(reports):
    metrics = {
        'class_0': {'precision': 0, 'recall': 0, 'f1-score': 0},
        'class_1': {'precision': 0, 'recall': 0, 'f1-score': 0},
        'accuracy': 0,
        'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0},
        'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0},
    }
    
    n_reports = len(reports)
    
    # Accumulate metrics
    for report in reports:
        for class_key in ['0', '1']:
            metrics[f'class_{class_key}']['precision'] += report[class_key]['precision']
            metrics[f'class_{class_key}']['recall'] += report[class_key]['recall']
            metrics[f'class_{class_key}']['f1-score'] += report[class_key]['f1-score']
        
        metrics['accuracy'] += report['accuracy']
        
        for avg_type in ['macro avg', 'weighted avg']:
            metrics[avg_type]['precision'] += report[avg_type]['precision']
            metrics[avg_type]['recall'] += report[avg_type]['recall']
            metrics[avg_type]['f1-score'] += report[avg_type]['f1-score']
    
    # Compute averages
    for key in metrics:
        if isinstance(metrics[key], dict):
            for metric in metrics[key]:
                metrics[key][metric] /= n_reports
        else:
            metrics[key] /= n_reports
    
    return metrics

def main():
    csv_path = 'NTU_datasets/dev_split_Depression_AVEC2017.csv'
    experiment_name = 'tan_train_tan_dev'
    seeds = [40, 41, 42, 43, 44]
    exclude_participants = [451, 458]
    
    # Load ground truth and convert to binary
    ground_truth_scores = load_ground_truth(csv_path, exclude_participants)
    ground_truth_binary = scores_to_binary(ground_truth_scores)
    
    # Evaluate across seeds
    reports = []
    for seed in seeds:
        predicted_binary = load_predictions(f'txt_ops/{experiment_name}_{seed}.txt')
        report = classification_report(ground_truth_binary, predicted_binary, digits=3, output_dict=True)
        reports.append(report)
    
    # Compute and display averaged metrics
    averaged_metrics = compute_averaged_metrics(reports)
    print("Averaged PHQ-8 Classification Metrics:")
    print(averaged_metrics)

if __name__ == "__main__":
    main()
