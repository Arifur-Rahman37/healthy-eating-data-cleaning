# Data cleaning for healthy_eating_dataset.csv using pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

	# Load cleaned data for visualization
	cleaned_df = pd.read_csv('healthy_eating_dataset_cleaned.csv')



	# Combined figure with five subplots
	fig, axes = plt.subplots(2, 3, figsize=(18, 10))



	# 1. Histogram of calories
	axes[0, 0].text(0.5, 1.18, 'Shows how calories are distributed across all meals.', ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=10)
	axes[0, 0].hist(cleaned_df['calories'], bins=30, color='skyblue', edgecolor='black')
	axes[0, 0].set_title('Calories Distribution')
	axes[0, 0].set_xlabel('Calories')
	axes[0, 0].set_ylabel('Frequency')
	# Decision description
	axes[0, 0].text(0.5, -0.25, 'Decision: Identify typical calorie ranges and spot outliers for meal planning.', ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=10, color='navy')


	# 2. Bar chart: average calories by diet type
	axes[0, 1].text(0.5, 1.18, 'Compares average calories for each diet type.', ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=10)
	avg_calories = cleaned_df.groupby('diet_type')['calories'].mean()
	avg_calories.plot(kind='bar', color='orange', edgecolor='black', ax=axes[0, 1])
	axes[0, 1].set_title('Avg Calories by Diet Type')
	axes[0, 1].set_xlabel('Diet Type')
	axes[0, 1].set_ylabel('Avg Calories')
	# Decision description
	axes[0, 1].text(0.5, -0.6, 'Decision: Choose diet types based on calorie goals (e.g., weight loss, maintenance).', ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=10, color='navy')


	# 3. Boxplot: calories by meal type
	axes[0, 2].text(0.5, 1.18, 'Shows calorie range and outliers for each meal type.', ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=10)
	cleaned_df.boxplot(column='calories', by='meal_type', ax=axes[0, 2], grid=False)
	axes[0, 2].set_title('Calories by Meal Type')
	axes[0, 2].set_xlabel('Meal Type')
	axes[0, 2].set_ylabel('Calories')
	axes[0, 2].figure.suptitle('')  # Remove automatic suptitle
	# Decision description
	axes[0, 2].text(0.5, -0.25, 'Decision: Select meal types to match calorie needs for different times of day.', ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=10, color='navy')


	# 4. Pie chart: proportion of meals by cuisine
	axes[1, 0].text(0.5, 1.18, 'Shows the proportion of meals for each cuisine.', ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=10)
	cuisine_counts = cleaned_df['cuisine'].value_counts()
	axes[1, 0].pie(cuisine_counts, labels=cuisine_counts.index, autopct='%1.1f%%', startangle=140)
	axes[1, 0].set_title('Meal Proportion by Cuisine')
	# Decision description
	axes[1, 0].text(0.5, -0.25, 'Decision: Adjust menu diversity or focus on popular cuisines for target audience.', ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=10, color='navy')


	# 5. Scatter plot: protein vs. calories
	axes[1, 1].text(0.5, 1.18, 'Shows relationship between protein and calories.', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)
	axes[1, 1].scatter(cleaned_df['calories'], cleaned_df['protein_g'], alpha=0.5, color='green')
	axes[1, 1].set_title('Protein vs. Calories')
	axes[1, 1].set_xlabel('Calories')
	axes[1, 1].set_ylabel('Protein (g)')
	# Decision description
	axes[1, 1].text(0.5, -0.25, 'Decision: Find high-protein, low-calorie meals for healthy recommendations.', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10, color='navy')


	# 6. Line graph: average calories by cuisine over meal_id
	axes[1, 2].text(0.5, 1.18, 'Shows trend of average calories by cuisine across meal IDs.', ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=10)
	for cuisine in cleaned_df['cuisine'].unique():
		subset = cleaned_df[cleaned_df['cuisine'] == cuisine]
		axes[1, 2].plot(subset['meal_id'], subset['calories'].rolling(window=10, min_periods=1).mean(), label=cuisine)
	axes[1, 2].set_title('Avg Calories by Cuisine (Meal ID Order)')
	axes[1, 2].set_xlabel('Meal ID')
	axes[1, 2].set_ylabel('Avg Calories (rolling mean)')
	axes[1, 2].legend(fontsize=8)
	# Decision description
	axes[1, 2].text(0.5, -0.25, 'Decision: Track calorie trends by cuisine to spot shifts or seasonal changes.', ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=10, color='navy')

	fig.suptitle('Healthy Eating Dataset Visualizations', fontsize=18, fontweight='bold')
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.show()
	print('First 5 rows of cleaned data:')
	print(pd.read_csv('healthy_eating_dataset_cleaned.csv').head())
	print('\nBasic statistics:')

