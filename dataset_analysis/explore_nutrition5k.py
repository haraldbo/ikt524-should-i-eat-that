"""
Script to explore and analyze the Nutrition5k dataset

This script helps understand the dataset structure and statistics
before training the nutrition prediction model.
It reads metadata files, computes statistics, and generates plots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# == Define the Nutrition5kExplorer class == #
class Nutrition5kExplorer:
    """Explore and analyze Nutrition5k dataset"""
    
    # == 1. Initialize with data directory
    def __init__(self, data_dir="./data/nutrition5k"):
        self.data_dir = Path(data_dir)
        self.metadata_dir = self.data_dir / "metadata"
        
        self.dish_data = None
        self.ingredient_data = None
        self.merged_data = None

    # == 2. Check for required files == #
    def check_files_exist(self):
        """Check if metadata files have been downloaded"""
        if not self.metadata_dir.exists():
            print("ERROR: Metadata directory not found!")
            print("Run: python models/download_nutrition5k.py --metadata-only")
            return False
        
        # Check for dish metadata files
        dish_files = list(self.metadata_dir.glob("dish_metadata_cafe*.csv"))
        ingredient_file = self.metadata_dir / "nutrition5k_dataset_metadata_ingredients_metadata.csv"
        
        # At least one dish metadata file must exist
        if not dish_files:
            print("ERROR: No dish metadata CSV files found!")
            return False
        
        print(f"Found {len(dish_files)} dish metadata file(s)")
        
        # Ingredient metadata is optional
        if ingredient_file.exists():
            print(f"Found ingredient metadata file")
        else:
            print(f"WARNING: Ingredient metadata not found (optional)")
        
        return True
    # == 3. Load and analyze data == #
    def load_ingredient_metadata(self):
        """Load ingredient metadata (nutritional info per ingredient)"""
        ingredient_file = self.metadata_dir / "nutrition5k_dataset_metadata_ingredients_metadata.csv"
        
        # Ingredient metadata is optional
        if not ingredient_file.exists():
            print("WARNING: Ingredient metadata file not found, skipping...")
            return None
        
        # Load ingredient metadata
        try:
            print("  Reading ingredient metadata...")
            df = pd.read_csv(ingredient_file)
            print(f"  Loaded {len(df)} ingredients")
            self.ingredient_data = df
            return df

        # Handle any errors during loading    
        except Exception as e:
            print(f"  WARNING: Error loading ingredient metadata: {e}")
            return None
    
    #== 4. Load dish metadata files == #
    def load_dish_data(self):
        """Load dish metadata from cafe CSV files"""
        print("\nLoading dish nutrition data...")
        
        # Ensure metadata directory exists
        all_dishes = []
        
        # Read each dish metadata CSV file
        for csv_file in sorted(self.metadata_dir.glob("dish_metadata_cafe*.csv")):
            print(f"  Reading {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, header=None, on_bad_lines='skip')
                
                if len(df.columns) < 7:
                    print(f"    WARNING: Skipping - insufficient columns")
                    continue
                
                # Extract dish-level data
                dish_df = df.iloc[:, :7].copy()
                dish_df.columns = [
                    'dish_id', 
                    'total_calories', 
                    'total_mass', 
                    'total_fat', 
                    'total_carb', 
                    'total_protein', 
                    'num_ingrs'
                ]
                
                # Convert numeric columns
                numeric_cols = ['total_calories', 'total_mass', 'total_fat', 
                               'total_carb', 'total_protein', 'num_ingrs']
                for col in numeric_cols:
                    dish_df[col] = pd.to_numeric(dish_df[col], errors='coerce')
                
                # Calculate actual ingredient counts
                ingredient_counts = []
                for idx, row in df.iterrows():
                    remaining_cols = len(row) - 7
                    if remaining_cols > 0:
                        num_ingredients = remaining_cols // 7
                        ingredient_counts.append(num_ingredients)
                    else:
                        ingredient_counts.append(0)
                
                # Add ingredient count and source cafe
                dish_df['actual_ingr_count'] = ingredient_counts
                dish_df['source_cafe'] = csv_file.stem.replace('dish_metadata_', '')
                
                all_dishes.append(dish_df)
                print(f"    Loaded {len(dish_df)} dishes")
            # Handle errors during file reading    
            except Exception as e:
                print(f"    WARNING: Error reading {csv_file.name}: {e}")
        # Combine all dish data 
        if not all_dishes:
            print("ERROR: Failed to load any dish data!")
            return None
        # Concatenate all dish dataframes
        self.dish_data = pd.concat(all_dishes, ignore_index=True)
        
        print(f"\nSuccessfully loaded {len(self.dish_data)} dishes total")
        print(f"Columns: {list(self.dish_data.columns)}")
        
        return self.dish_data
    # == 5. Merge ingredient counts with dish data == #
    def merge_ingredient_counts(self):
        """Merge ingredient counts with dish data"""
        if self.dish_data is None:
            self.load_dish_data()
        
        if self.ingredient_data is not None:
            print("\nMerging ingredient data with dishes...")
            self.merged_data = self.dish_data.copy()
            self.merged_data['has_ingredient_metadata'] = True
            print(f"Merged data ready with {len(self.merged_data)} dishes")
        else:
            self.merged_data = self.dish_data.copy()
            self.merged_data['has_ingredient_metadata'] = False

    # == 6. Analyze nutritional statistics == #
    def analyze_nutrition_stats(self):
        """Analyze nutritional value statistics"""
        if self.dish_data is None:
            self.load_dish_data()
        
        if self.dish_data is None:
            return None
        
        print("\nNutritional Statistics:")
        print("=" * 70)
        
        # Key nutritional columns
        nutrition_cols = ['total_calories', 'total_mass', 'total_fat', 
                         'total_carb', 'total_protein']
        
        stats = self.dish_data[nutrition_cols].describe()
        stats_formatted = stats.round(2)
        print(stats_formatted.to_string())
        
        print("\nMissing Values:")
        missing = self.dish_data[nutrition_cols].isnull().sum()
        if missing.sum() == 0:
            print("  No missing values - dataset is complete!")
        else:
            print(missing.to_string())
        
        print("\nAdditional Insights:")
        print(f"  Average calories per dish: {self.dish_data['total_calories'].mean():.2f} kcal")
        print(f"  Average mass per dish: {self.dish_data['total_mass'].mean():.2f} g")
        print(f"  Average protein: {self.dish_data['total_protein'].mean():.2f} g")
        print(f"  Average carbs: {self.dish_data['total_carb'].mean():.2f} g")
        print(f"  Average fat: {self.dish_data['total_fat'].mean():.2f} g")
        
        return stats
    
    # == 7. Analyze food categories based on ingredient counts == #
    def get_food_categories(self):
        """Analyze number of ingredients per dish"""
        if self.merged_data is None:
            self.merge_ingredient_counts()
        
        if self.merged_data is None:
            return
        
        print("\nDish Complexity (Ingredient Count):")
        print("=" * 70)
        
        # Prefer actual ingredient count if available
        ingr_col = 'actual_ingr_count' if 'actual_ingr_count' in self.merged_data.columns else 'num_ingrs'
        
        if ingr_col not in self.merged_data.columns:
            print("  WARNING: No ingredient count data available")
            return
        
        # Filter out invalid counts
        valid_data = self.merged_data[ingr_col].dropna()
        valid_data = valid_data[valid_data > 0]
        
        if len(valid_data) == 0:
            print("  WARNING: No valid ingredient count data available")
            
            if 'num_ingrs' in self.merged_data.columns:
                print("  Trying num_ingrs column as fallback...")
                valid_data = self.merged_data['num_ingrs'].dropna()
                valid_data = valid_data[valid_data > 0]
        
        if len(valid_data) > 0:
            avg_ingrs = valid_data.mean()
            min_ingrs = valid_data.min()
            max_ingrs = valid_data.max()
            
            print(f"  Dishes with ingredient data: {len(valid_data)}")
            print(f"  Average ingredients per dish: {avg_ingrs:.1f}")
            print(f"  Min ingredients: {int(min_ingrs)}")
            print(f"  Max ingredients: {int(max_ingrs)}")
            
            print("\n  Distribution of dishes by ingredient count:")
            counts = valid_data.value_counts().sort_index()
            
            for ingr_count, num_dishes in counts.items():
                bar_length = min(int(num_dishes / 50), 60)
                bar = '#' * bar_length
                print(f"    {int(ingr_count):2d} ingredients: {num_dishes:4d} dishes {bar}")
        else:
            print("  WARNING: Could not determine ingredient counts")
            print(f"  Available columns: {list(self.merged_data.columns)}")
    
    # == 8. Analyze ingredient metadata if available == #
    def analyze_ingredients(self):
        """Analyze ingredient metadata if available"""
        if self.ingredient_data is None:
            return
        
        print("\nIngredient Database Analysis:")
        print("=" * 70)
        
        print(f"  Total unique ingredients: {len(self.ingredient_data)}")
        print(f"  Columns: {list(self.ingredient_data.columns)}")
        
        print("\n  Sample ingredients:")
        sample_cols = min(len(self.ingredient_data.columns), 6)
        print(self.ingredient_data.iloc[:10, :sample_cols].to_string(index=False))
    
    # == 9. Plot distributions of nutritional values == #
    def plot_distributions(self, save=True):
        """Plot distribution of nutritional values"""
        if self.dish_data is None:
            self.load_dish_data()
        
        if self.dish_data is None:
            return
        
        # Set up the plotting area
        fig = plt.figure(figsize=(18, 13))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, top=0.95, bottom=0.05)
        
        fig.suptitle('Nutrition5k Dataset - Comprehensive Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Define nutritional columns and their plot settings
        nutrition_cols = {
            'total_calories': ('Calories (kcal)', 'blue', 0),
            'total_mass': ('Mass (g)', 'green', 1),
            'total_fat': ('Fat (g)', 'red', 2),
            'total_carb': ('Carbohydrates (g)', 'orange', 3),
            'total_protein': ('Protein (g)', 'purple', 4)
        }
        
        # Plot histograms for each nutritional value
        for col, (label, color, idx) in nutrition_cols.items():
            if idx < 3:
                ax = fig.add_subplot(gs[0, idx])
            else:
                ax = fig.add_subplot(gs[1, idx - 3])
            
            data = self.dish_data[col].dropna()
            
            ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color=color)
            ax.set_title(label, fontweight='bold', fontsize=11)
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.grid(alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=8)
            
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                      label=f'Mean: {mean_val:.1f}')
            ax.legend(fontsize=8)
        
        # Plot ingredient count distribution if available
        if 'actual_ingr_count' in self.dish_data.columns:
            ax = fig.add_subplot(gs[1, 2])
            ingr_data = self.dish_data['actual_ingr_count'].dropna()
            ingr_data = ingr_data[ingr_data > 0]
            
            ax.hist(ingr_data, bins=range(int(ingr_data.min()), int(ingr_data.max())+2), 
                   edgecolor='black', alpha=0.7, color='teal')
            ax.set_title('Ingredient Count Distribution', fontweight='bold', fontsize=11)
            ax.set_xlabel('Number of Ingredients', fontsize=9)
            ax.set_ylabel('Number of Dishes', fontsize=9)
            ax.grid(alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=8)
            
            mean_ingr = ingr_data.mean()
            ax.axvline(mean_ingr, color='red', linestyle='dashed', linewidth=2,
                      label=f'Mean: {mean_ingr:.1f}')
            ax.legend(fontsize=8)
        # Plot additional insights
        if 'source_cafe' in self.dish_data.columns:
            ax = fig.add_subplot(gs[2, 0])
            cafe_counts = self.dish_data['source_cafe'].value_counts()
            
            ax.bar(range(len(cafe_counts)), cafe_counts.values, color='skyblue', edgecolor='black')
            ax.set_xticks(range(len(cafe_counts)))
            ax.set_xticklabels(cafe_counts.index, rotation=45, ha='right', fontsize=8)
            ax.set_title('Dishes by Source Cafe', fontweight='bold', fontsize=11)
            ax.set_xlabel('Cafe', fontsize=9)
            ax.set_ylabel('Number of Dishes', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3, axis='y', linestyle='--')
        
        # Scatter plot of calories vs protein
        ax = fig.add_subplot(gs[2, 1])
        ax.scatter(self.dish_data['total_protein'], self.dish_data['total_calories'], 
                  alpha=0.5, s=10, color='darkgreen')
        ax.set_xlabel('Protein (g)', fontweight='bold', fontsize=9)
        ax.set_ylabel('Calories (kcal)', fontweight='bold', fontsize=9)
        ax.set_title('Calories vs Protein', fontweight='bold', fontsize=11)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Pie chart of average macronutrient composition
        ax = fig.add_subplot(gs[2, 2])
        avg_fat = self.dish_data['total_fat'].mean()
        avg_carb = self.dish_data['total_carb'].mean()
        avg_protein = self.dish_data['total_protein'].mean()
        
        # Pie chart
        sizes = [avg_fat, avg_carb, avg_protein]
        labels = [f'Fat\n{avg_fat:.1f}g', f'Carbs\n{avg_carb:.1f}g', f'Protein\n{avg_protein:.1f}g']
        colors = ['red', 'orange', 'purple']
        
        # Create pie chart
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, textprops={'fontsize': 9})
        ax.set_title('Average Macronutrient\nComposition', fontweight='bold', fontsize=11)
        
        # Save the figure if required
        if save:
            output_path = self.data_dir / "nutrition_comprehensive_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
        
        plt.show()
    
    # == 10. Sample dishes with nutrition info == #
    def sample_dishes(self, n=10):
        """Show sample dishes with their nutrition info"""

        # Ensure dish data is loaded
        if self.dish_data is None:
            self.load_dish_data()
        
        # If no data, return
        if self.dish_data is None:
            return
        
        print(f"\nSample of {n} Random Dishes:")
        print("=" * 70)
        
        # Select relevant columns to display
        cols_to_show = ['dish_id', 'total_calories', 'total_protein', 
                       'total_carb', 'total_fat', 'total_mass']
        
        # Include ingredient count if available
        if 'actual_ingr_count' in self.dish_data.columns:
            cols_to_show.append('actual_ingr_count')
        elif 'num_ingrs' in self.dish_data.columns:
            cols_to_show.append('num_ingrs')
        
        # Include source cafe if available
        sample = self.dish_data[cols_to_show].sample(min(n, len(self.dish_data)))
        # Format display options for better readability
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.2f}'.format)
        
        print(sample.to_string(index=False))
    
    # == 11. Save cleaned dataset for training == #
    def save_cleaned_dataset(self):
        """Save a cleaned version of the dataset for training"""
        # Ensure dish data is loaded
        if self.dish_data is None:
            return
        # If no data, return
        output_path = self.data_dir / "cleaned_dish_metadata.csv"
        # Select key columns to save
        cols_to_save = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 
                       'total_carb', 'total_protein']
        # Include ingredient count if available
        if 'actual_ingr_count' in self.dish_data.columns:
            cols_to_save.append('actual_ingr_count')
        # Include source cafe if available
        if 'source_cafe' in self.dish_data.columns:
            cols_to_save.append('source_cafe')
        # Filter out rows with missing nutritional data
        clean_df = self.dish_data[cols_to_save].dropna()
        # Save to CSV
        clean_df.to_csv(output_path, index=False)
        print(f"\nSaved cleaned dataset to: {output_path}")
        print(f"  Contains {len(clean_df)} dishes with complete nutritional data")
        
        print(f"\n  Data merged from:")
        if 'source_cafe' in clean_df.columns:
            for cafe in clean_df['source_cafe'].unique():
                count = len(clean_df[clean_df['source_cafe'] == cafe])
                print(f"    - dish_metadata_{cafe}.csv: {count} dishes")
        
        if self.ingredient_data is not None:
            print(f"    - nutrition5k_dataset_metadata_ingredients_metadata.csv: {len(self.ingredient_data)} ingredients")
            print(f"  Note: Ingredient metadata available for enhanced analysis")
        
        return clean_df
    
    # == 12. Generate a complete summary report == #
    def generate_summary_report(self):
        """Generate a complete summary report"""
        print("\n" + "=" * 70)
        print(" " * 15 + "NUTRITION5K DATASET EXPLORATION REPORT")
        print("=" * 70)
        # Check for required files
        if not self.check_files_exist():
            return
        # Load and analyze data
        self.load_dish_data()
        self.load_ingredient_metadata()
        self.merge_ingredient_counts()
        # If no dish data, cannot proceed
        if self.dish_data is None:
            print("\nERROR: Failed to load dish data. Cannot generate report.")
            return
        # Perform analyses
        self.analyze_nutrition_stats()
        self.get_food_categories()
        # Analyze ingredient metadata if available
        if self.ingredient_data is not None:
            self.analyze_ingredients()
        # Sample some dishes
        self.sample_dishes()
        clean_df = self.save_cleaned_dataset()
        
        print("\n" + "=" * 70)
        print("Dataset Preparation Complete")
        print("\n  - Cleaned metadata CSV has been generated")
        print("  - Nutritional distributions have been analyzed")
        print("  - Dataset is ready for model training")
        print("=" * 70)

# == Main execution == #
def main():
    # Create an explorer instance and generate report
    explorer = Nutrition5kExplorer()
    explorer.generate_summary_report()
    # Optionally generate plots
    if explorer.dish_data is not None:
        try:
            print()
            response = input("Generate comprehensive analysis plots? (y/n): ")
            if response.lower() == 'y':
                explorer.plot_distributions()
        except KeyboardInterrupt:
            print("\nSkipping plots.")
    else:
        print("\nWARNING: Cannot generate plots - no data loaded.")


if __name__ == "__main__":
    main()