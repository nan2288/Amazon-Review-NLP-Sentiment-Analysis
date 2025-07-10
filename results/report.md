# MSE 641 Assignment 4 Report

**Student Name:** NAN WANG
**Student ID:** 21143092(N96WANG)
**Date:** 2025-6-16

*Maximum 10 sentences total for all analysis sections below.*

## Final Results Summary

| Activation Function | Best L2 Regularization | Best Dropout Rate | Validation Accuracy | Test Accuracy |
|---------------------|------------------------|-------------------|---------------------|---------------|
| ReLU                | [0.001]               | [0.3]           | [0.74535]        | [0.74635] |
| Sigmoid             | [0.001]               | [0.3]           | [0.73625]        | [0.720325] |
| Tanh                | [0.001]               | [0.3]           | [0.7415625]        | [0.741125] |

## Analysis (Maximum 10 sentences total)

### Effect of Activation Functions (2-3 sentences)
*Which activation function performed best and why do you think that is?*

ReLU achieved the highest validation (0.74535) and test (0.74635) accuracy, demonstrating superior performance due to its efficient gradient propagation and avoidance of saturation. Tanh showed intermediate performance with stable results, while Sigmoid performed worst due to its vanishing gradient issues in deeper layers.


### Effect of L2 Regularization (2-3 sentences)
*How did L2 regularization affect your model performance? Did it help prevent overfitting?*

Lower L2 values (0.001) consistently outperformed higher regularization (0.01), providing a significant average 10.2% accuracy advantage. Excessive L2=0.01 regularization severely degraded Sigmoid performance by 16.1%, indicating excessive parameter shrinkage that harmed model capacity.


### Effect of Dropout (2-3 sentences)
*How did dropout affect your results? What dropout rates worked best?*

A 0.3 dropout rate proved optimal across all activations, consistently outperforming 0.5 dropout by over 2.8% average accuracy. Higher dropout rates caused excessive information loss, particularly hurting complex patterns where Sigmod showed a dramatic 22.7% accuracy drop.


### Best Configuration and Key Insight (2-3 sentences)
*What was your best overall model and why do you think this combination worked well? What was the most important thing you learned from this assignment?*

**Best Model:** [ReLU] with L2=[0.001], Dropout=[0.3], Test Accuracy=[0.74635]
This combination succeeded by leveraging ReLU's gradient efficiency balanced with sufficient regularization capacity. The critical insight reveals that activation function choice has greater impact than hyperparameter tuning, with ReLU's superior gradient flow providing the fundamental advantage.
