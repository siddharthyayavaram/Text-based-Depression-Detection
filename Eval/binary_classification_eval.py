import pandas as pd
from sklearn.metrics import classification_report

def load_ground_truth(csv_path, exclude_participants=None):
    df = pd.read_csv(csv_path)
    if exclude_participants:
        df.drop(df[df["Participant_ID"].isin(exclude_participants)].index, inplace=True)
    return df['PHQ8_Binary'].tolist()

def load_predictions(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    predictions = content.split("Participant_ID: \nResponse: ")[1:]
    predictions = [pred.strip() for pred in predictions]
    return [1 if pred == 'Depressed' else 0 for pred in predictions]

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
    # Configuration
    csv_path = 'NTU_datasets/dev_split_Depression_AVEC2017.csv'
    experiment_name = 'tan_bin_tan_dev_phq'
    seeds = [40, 42, 44]
    exclude_participants = [451, 458]
    
    # Load ground truth
    ground_truth = load_ground_truth(csv_path, exclude_participants)
    
    # Evaluate across seeds
    reports = []
    for seed in seeds:
        predictions = load_predictions(f'txt_ops/{experiment_name}_{seed}.txt')
        report = classification_report(ground_truth, predictions, digits=3, output_dict=True)
        reports.append(report)
    
    # Compute and display averaged metrics
    averaged_metrics = compute_averaged_metrics(reports)
    print("Averaged Classification Metrics:")
    print(averaged_metrics)

if __name__ == "__main__":
    main()