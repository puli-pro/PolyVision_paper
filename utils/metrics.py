from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
app=FastAPI()
class Student(BaseModel):
    id: int
    name: str
    age: int
student_DB=[]
@app.get("/students")
async def get_students():
    return student_DB

@app.post("/students")
async def add_students(student:Stude# --- file: utils/metrics.py ---
"""
Evaluation metrics for PolyVision framework
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class MetricsCalculator:
    """Utility class for calculating various evaluation metrics"""
    
    @staticmethod
    def calculate_auroc_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """Calculate AUROC and AUPRC scores"""
        auroc = roc_auc_score(y_true, y_scores)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = auc(recall, precision)
        
        return auroc, auprc
    
    @staticmethod
    def calculate_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Calculate sensitivity and specificity"""
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensitivity = specificity = 0.0
            
        return sensitivity, specificity
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_probs: np.ndarray, 
                                 n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

class CrossValidationResults:
    """Class to store and manage cross-validation results"""
    
    def __init__(self):
        self.fold_results = []
        
    def add_fold_result(self, fold_idx: int, metrics: Dict[str, float]):
        """Add results from a single fold"""
        result = {'fold': fold_idx}
        result.update(metrics)
        self.fold_results.append(result)
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics across all folds"""
        if not self.fold_results:
            return {}
            
        metrics = list(self.fold_results[0].keys())
        metrics.remove('fold')  # Remove fold index
        
        summary = {}
        for metric in metrics:
            values = [fold[metric] for fold in self.fold_results]
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
            
        return summarynt):
    student_DB.append(student.dict())
    return {"msg": "Student added successfully"}

@app.put("/students/{student_id}")
async def update(student_id : int , student : Student):
    for s in student_DB:
        if s["id"]==student_id:
            s.update(student.dict())
            return {"msg":"Student details updated successfully"}
    raise HTTPException(status_code=404,detail="Student not found")

@app.delete("/students/{student_id}")
async def delete_student(student_id: int, student : Student):
     for s in student_DB:
        if s["id"] == student_id:
            student_DB.remove(s)
            return {"msg": "Student deleted"}
     raise HTTPException(status_code=404, detail="Student not found")