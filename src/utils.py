import pandas as pd
from pathlib import Path
import os

class SaveModelMetrics:
    def __init__(self, file_path) -> None:
        self.file_path = Path(os.path.expanduser(file_path))

    def save_metrics(self, name: str, description: str, algorithm: str, best_params_grid_search: dict, params: dict, accuracy_score: float, precision_score: float, recall: float, f1_score: float, conf_matrix: list) -> None:
        data = {
            'model_name': [name],
            'description': [description],
            'algorithm': [algorithm],
            'best_params_grid_search': [best_params_grid_search],
            'params': [params],
            'accuracy_score': [accuracy_score],
            'precision_score': [precision_score],
            'recall': [recall],
            'f1_score': [f1_score],
            'confusion_matrix': [conf_matrix]
        }

        df = pd.DataFrame(data)

        if not self.file_path.exists():
            df.to_csv(self.file_path, index=False)
        else:
            with open(self.file_path, 'a') as f:
                df.to_csv(f, header=False, index=False)
