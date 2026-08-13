# Andrew Ng — ML Specialization: Course 1, Week 1

## Key Concepts

### Supervised Learning
Supervised learning is the process of training a model by showing it 
inputs paired with correct outputs (labels). The model learns by 
comparing its predictions to the labels and adjusting to reduce the 
error. The labels are the teacher during training — verification 
happens later on unseen data.

### Regression vs Classification
- **Regression:** Predicting continuous values (how much?)
- **Classification:** Predicting discrete categories (which one?)

## Examples from My Domain

### Regression
Predicting PM2.5 air pollution levels from sensor data across 
Nairobi stations — the output is a continuous number.

### Classification
Classifying each day's air quality as Good, Moderate, or Unhealthy 
based on PM2.5 thresholds — the output is a category.

## Connections to My Work
My `predict()` function in `linear_regression.py` is doing regression.
If I wanted classification, I would need a different function 
(logistic regression) that outputs probabilities for each category.

## Questions to Explore
- How does logistic regression turn a continuous output into a 
  category?
- What happens when categories are imbalanced (most days are 
  "Moderate")?