

# Data cleaning for healthy_eating_dataset.csv using pandas and numpy
import pandas as pd
import numpy as np

def clean_and_preview_csv(input_path, output_path):
	# Load the dataset
	df = pd.read_csv(input_path)

	# Drop duplicate rows
	df = df.drop_duplicates()

	# Remove rows with missing critical values (e.g., meal_id, meal_name, calories)
	df = df.dropna(subset=['meal_id', 'meal_name', 'calories'])

	# Convert numeric columns to appropriate types
	numeric_cols = ['calories', 'protein_g', 'carbs_g', 'fat_g', 'fiber_g', 'sugar_g', 'sodium_mg', 'cholesterol_mg', 'serving_size_g', 'prep_time_min', 'cook_time_min', 'rating']
	for col in numeric_cols:
		df[col] = pd.to_numeric(df[col], errors='coerce')

	# Fill missing values in non-critical columns with median (for numeric) or mode (for categorical)
	for col in numeric_cols:
		if df[col].isnull().any():
			df[col].fillna(df[col].median(), inplace=True)

	categorical_cols = ['cuisine', 'meal_type', 'diet_type', 'cooking_method', 'is_healthy']
	for col in categorical_cols:
		if col in df.columns and df[col].isnull().any():
			df[col].fillna(df[col].mode()[0], inplace=True)

	# Remove outliers in calories (outside 1st and 99th percentile)
	q_low = df['calories'].quantile(0.01)
	q_high = df['calories'].quantile(0.99)
	df = df[(df['calories'] >= q_low) & (df['calories'] <= q_high)]

	# Save cleaned data
	df.to_csv(output_path, index=False)


if __name__ == "__main__":
	clean_and_preview_csv('healthy_eating_dataset.csv', 'healthy_eating_dataset_cleaned.csv')
	print('First 5 rows of cleaned data:')
	print(pd.read_csv('healthy_eating_dataset_cleaned.csv').head())
	print('\nBasic statistics:')

