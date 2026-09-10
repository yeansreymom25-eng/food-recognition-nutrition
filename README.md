# Food Recognition and Nutritional Estimation Using Deep Learning

## Project Overview

This project is a deep learning system that recognizes food from an image and provides estimated nutritional information.

The system uses PyTorch and MobileNetV2 with transfer learning.

After predicting the food, the system displays:

- Food name
- Food category
- Confidence score
- Portion size
- Calories
- Protein
- Carbohydrates
- Fat

If the prediction confidence is below 50%, the system shows "Uncertain Prediction" and does not display nutrition information.

---

## Model

- Framework: PyTorch
- Model: MobileNetV2
- Method: Transfer Learning and Fine-Tuning
- Final Model: Improved Fine-Tuned MobileNetV2
- Number of Classes: 20
- Number of Categories: 5
- Confidence Threshold: 50%

---

## Food Categories

### Fast Food
- Pizza
- Hamburger
- French Fries
- Hot Dog

### Fruit
- Apple
- Banana
- Orange
- Strawberry

### Vegetable
- Carrot
- Broccoli
- Tomato
- Cucumber

### Nuts
- Almond
- Peanut
- Cashew
- Walnut

### Dessert
- Donut
- Ice Cream
- Chocolate Cake
- Cupcake

---

## Dataset

The project uses public food datasets and manually collected internet images.

The final training dataset contains:

- Training: 2,151 images
- Validation: 596 images
- Testing: 311 images
- Classes: 20

Additional diverse images were added to improve weak food classes.

---

## Model Performance

### Original MobileNetV2
Test Accuracy: 92.28%

### Fine-Tuned MobileNetV2
Test Accuracy: 95.18%

### Final Improved MobileNetV2
Test Accuracy: 95.50%

Correct Predictions: 297 / 311

The final improved model was selected as the final model.

---

## System Workflow

1. User uploads a food image.
2. The image is resized to 224 × 224.
3. The image is normalized.
4. MobileNetV2 predicts the food class.
5. The system calculates the confidence score.
6. If confidence is below 50%, the result is shown as uncertain.
7. If confidence is at least 50%, the food category is retrieved.
8. The user enters the estimated portion size in grams.
9. The system calculates estimated nutrition.
10. The final result is displayed.

---

## Nutrition Estimation

The nutrition database contains:

- Calories
- Protein
- Carbohydrates
- Fat

Nutrition values are based on approximately 100g of food.

For other portion sizes:

Nutrition = Nutrition per 100g × Portion Weight / 100

The nutrition values are estimates because the exact weight and ingredients cannot be determined from one normal image.

---

## Final Model

Final model file:

`best_improved_mobilenetv2.pth`

Final Test Accuracy:

**95.50%**

---

## Tools

- Python
- PyTorch
- torchvision
- MobileNetV2
- Google Colab
- Google Drive
- GitHub
- Pandas
- Matplotlib
- Scikit-learn
- PIL
