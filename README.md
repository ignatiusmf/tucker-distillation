# Tucker-Based Knowledge Distillation for CNNs
This readme is currently generated with AI and will be refined at a later date.
## Synopsis

This project explores novel strategies for knowledge distillation in convolutional neural networks (CNNs), focusing on feature map compression using Tucker decomposition. It implements and benchmarks multiple distillation variants on CIFAR-10 and CIFAR-100 datasets using ResNet-based teacher and student models.

The work includes both direct feature map distillation and advanced tensor decomposition techniques to improve student performance while maintaining computational efficiency. The experiments also compare decomposed feature reconstruction effects from different network components (teacher, student, or both).

---

## Deeper Explanation

Knowledge distillation allows a smaller "student" model to learn from a larger "teacher" model. In this project, we go beyond standard output mimicking and instead target intermediate feature maps. Specifically, we:

- Apply **feature map distillation**, where L1 loss is used on intermediate activations.
- Extend this with **Tucker decomposition** of feature maps, leveraging tensor factorization to project and compress features.
- Experiment with **Tucker recomposition**, which reconstructs features from either the teacher, student, or both using shared factor matrices, then minimizes reconstruction error.

All experiments are automated and repeatable with SLURM job submission scripts and flexible command-line arguments. Metrics, losses, and accuracies are saved and plotted across training epochs.

---

## Project Structure

```
|-- analyzer_results_aggregator.ipynb    
|-- analyzer_tucker_distillation.ipynb   
|-- tucker_distillation.py                
|-- runner.py                            
|-- run.job                              
|-- util.sh                             
|-- test_noise.py                       
|-- toolbox/
|   |-- data_loader.py                  
|   |-- models.py                       
|   |-- utils.py                        
|-- experiments/                         
|-- data/                               
```

---

## Experiment Variants and Details

### 1. **Feature Map Distillation (Baseline)**
The base distillation method uses direct L1 loss on normalized intermediate feature maps between teacher and student networks, in addition to the standard classification loss.

### 2. **Tucker Distillation**
Decomposes teacher feature maps using Tucker decomposition and transfers compressed core tensors to the student. Student features are projected using teacher factors, and L1 loss is applied between the resulting cores.

### 3. **Tucker Recomposed Distillation**
Explores three recomposition strategies:
- **Teacher**: Only the teacher’s features are decomposed and recomposed.
- **Student**: Student's features are projected into the teacher's tensor space and reconstructed.
- **Both**: Both teacher and student features are decomposed and reconstructed for loss computation.

Each experiment is configurable via CLI flags:
```bash
--distillation [featuremap | tucker | tucker_recomp]
--ranks BATCH_SIZE,32,8,8  # Tensor ranks
--recomp_target [teacher | student | both]
```

### 4. **Automation and Scaling**
`runner.py` orchestrates all experiment launches across SLURM using `run.job` template and `generate_pbs_script(...)` logic. It dynamically skips existing runs and checks queue status to limit job submissions.

### 5. **Metrics and Visualization**
Each run saves:
- `metrics.json`: Train/test loss and accuracy logs
- `Loss.png`, `Accuracy.png`: Plots of learning trends
- `ResNet56.pth`: Best student checkpoint

These are aggregated using `analyzer_results_aggregator.ipynb` for comparison.

---
