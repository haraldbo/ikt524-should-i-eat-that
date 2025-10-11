"""
The following script explores the Nutrition5k dataset.

It does the following:
- Checks if metadata files are downloaded
- Loads and combines metadata from multiple CSVs
- Analyzes nutritional value statistics
- Plots distributions of calories, protein, carbs, fat, mass
- Examines number of ingredients per dish
- Samples random dishes with nutrition info
- Generates a summary report

This script is meant to help better understand the dataset structure 
and statistics before training the nutrition prediction model.

It assumes the metadata CSVs have been downloaded using the
`download_nutrition5k.py` script.
source: https://github.com/google-research-datasets/Nutrition5k?tab=readme-ov-file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# == 1. Nutrition5k Explorer == #
class Nutrition5kExplorer:
    """Explore and analyze Nutrition5k dataset"""
    
    def __init__(self, data_dir="./data/nutrition5k"):
        self.data_dir = Path(data_dir)
        self.metadata_dir = self.data_dir / "metadata"
        
        self.data = None
    # == 2. Check Files Exist == #
    def check_files_exist(self):
        """Check if metadata files have been downloaded"""
        if not self.metadata_dir.exists():
            print("Metadata directory not found!")
            print("Run: python models/download_nutrition5k.py --metadata-only")
            return False
        # Check for expected CSV files
        csv_files = list(self.metadata_dir.glob("dish_metadata_cafe*.csv"))
        if not csv_files:
            print("No dish metadata CSV files found!")
            return False
        
        print(f"Found {len(csv_files)} metadata file(s)!")
        return True
    
    # == 3. Load and Combine Data == #
    def load_data(self):
        """Load and combine metadata from all cafes"""
        print("\nLoading nutrition data...")
        
        # Check files first
        all_dataframes = []
        for csv_file in sorted(self.metadata_dir.glob("dish_metadata_cafe*.csv")):
            print(f"  Reading {csv_file.name}...")
            try:
                # Read CSV - the real format has no header row, just data
                df = pd.read_csv(csv_file, header=None, on_bad_lines='skip')
                
                # The first few columns are the main dish data:
                # Column 0: dish_id
                # Column 1: total_calories
                # Column 2: total_mass
                # Column 3: total_fat
                # Column 4: total_carb
                # Column 5: total_protein
                # Column 6: num_ingredients
                # Remaining columns: ingredient data (repeating pattern)
                
                # Extract main nutrition columns
                if len(df.columns) >= 7:
                    nutrition_df = df.iloc[:, :7].copy()
                    nutrition_df.columns = [
                        'dish_id', 'total_calories', 'total_mass', 
                        'total_fat', 'total_carb', 'total_protein', 'num_ingrs'
                    ]
                    all_dataframes.append(nutrition_df)
                else:
                    print(f"Skipping {csv_file.name} - unexpected format")
                    
            except Exception as e:
                print(f"Error reading {csv_file.name}: {e}")
        
        if not all_dataframes:
            print("Failed to load any data!")
            return None
        
        # Combine all dataframes
        self.data = pd.concat(all_dataframes, ignore_index=True)
        
        # Convert numeric columns
        numeric_cols = ['total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein', 'num_ingrs']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        print(f"\nSuccessfully loaded {len(self.data)} dishes")
        print(f"Columns: {list(self.data.columns)}")
        
        return self.data
    
    # == 4. Analyze Nutrition Stats == #
    def analyze_nutrition_stats(self):
        """Analyze nutritional value statistics"""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return None
        
        print("\nNutritional Statistics:")
        print("=" * 70)
        
        # Basic stats for each nutrition column
        nutrition_cols = ['total_calories', 'total_mass', 'total_fat', 
                         'total_carb', 'total_protein']
        
        # Summary statistics (table)
        stats = self.data[nutrition_cols].describe()
        print(stats.to_string())
        
        # Check for missing values
        print("\n Missing Values:")
        missing = self.data[nutrition_cols].isnull().sum()
        print(missing.to_string())
        
        # Additional insights
        print("\nAdditional Insights:")
        print(f"  Average calories per dish: {self.data['total_calories'].mean():.1f} kcal")
        print(f"  Average mass per dish: {self.data['total_mass'].mean():.1f} g")
        print(f"  Average protein: {self.data['total_protein'].mean():.1f} g")
        print(f"  Average carbs: {self.data['total_carb'].mean():.1f} g")
        print(f"  Average fat: {self.data['total_fat'].mean():.1f} g")
        
        return stats
    
    # == 5. Plot Distributions == #
    def plot_distributions(self, save=True):
        """Plot distribution of nutritional values"""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return
        
        # Plot histograms for each nutrition column
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Nutrition5k Dataset - Nutritional Value Distributions', 
                     fontsize=16, fontweight='bold')
        
        # Define colors and labels
        nutrition_cols = {
            'total_calories': ('Calories (kcal)', 'blue'),
            'total_mass': ('Mass (g)', 'green'),
            'total_fat': ('Fat (g)', 'red'),
            'total_carb': ('Carbohydrates (g)', 'orange'),
            'total_protein': ('Protein (g)', 'purple')
        }
        
        # Plot each distribution in a subplot 
        for idx, (col, (label, color)) in enumerate(nutrition_cols.items()):
            ax = axes[idx // 3, idx % 3]
            data = self.data[col].dropna()
            
            # Plot histogram with mean line and grid
            ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color=color)
            ax.set_title(label, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3, linestyle='--')
            
            # Add mean line
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                      label=f'Mean: {mean_val:.1f}')
            ax.legend()
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        # Save plot if desired
        if save:
            output_path = self.data_dir / "nutrition_distributions.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
        
        plt.show()
    
    # == 6. Food Categories and Ingredients == #
    def get_food_categories(self):
        """Analyze number of ingredients per dish"""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return
        
        print("\nDish Complexity (Ingredient Count):")
        print("=" * 70)
        
        # Ingredient count stats 
        if 'num_ingrs' in self.data.columns:
            # Remove NaN values before calculating stats
            valid_data = self.data['num_ingrs'].dropna()
            
            # Check if there's any valid data to avoid errors on empty series
            if len(valid_data) == 0:
                print("No valid ingredient count data available")
                return
            
            # Basic stats
            avg_ingrs = valid_data.mean()
            min_ingrs = valid_data.min()
            max_ingrs = valid_data.max()
            
            print(f"  Average ingredients per dish: {avg_ingrs:.1f}")
            print(f"  Min ingredients: {int(min_ingrs)}")
            print(f"  Max ingredients: {int(max_ingrs)}")
            
            print("\n  Distribution of dishes by ingredient count:")

            # Show counts for ingredient counts up to 15 for brevity 
            counts = valid_data.value_counts().sort_index().head(15)
            for ingr_count, num_dishes in counts.items():
                print(f"    {int(ingr_count):2d} ingredients: {num_dishes:4d} dishes")

    # == 7. Sample Dishes == #
    def sample_dishes(self, n=10):
        """Show sample dishes with their nutrition info"""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return
        
        print(f"\nSample of {n} Random Dishes:")
        print("=" * 70)
        
        # Select relevant columns to display and sample n rows 
        cols_to_show = ['dish_id', 'total_calories', 'total_protein', 
                       'total_carb', 'total_fat', 'total_mass', 'num_ingrs']
        
        sample = self.data[cols_to_show].sample(min(n, len(self.data)))
        
        # Format the display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.1f}'.format)
        
        print(sample.to_string(index=False))

    # == 8. Generate Summary Report == #
    def generate_summary_report(self):
        """Generate a complete summary report"""
        print("\n" + "=" * 70)
        print(" " * 15 + "NUTRITION5K DATASET EXPLORATION REPORT")
        print("=" * 70)
        
        # Check files exist 
        if not self.check_files_exist():
            return
        
        # Load data and analyze
        self.load_data()
        
        if self.data is None:
            print("\n Failed to load data. Cannot generate report.")
            return
        
        # Run analyses and print results
        self.analyze_nutrition_stats()
        self.get_food_categories()
        self.sample_dishes()
        
        print("\n" + "=" * 70)
        print("Exploration complete!")
        print("=" * 70)

        # Optionally save cleaned combined dataset
        output_path = self.data_dir / "cleaned_dish_metadata.csv"
        self.data.to_csv(output_path, index=False)
        print(f"\nSaved cleaned dataset to: {output_path}")


# == Main Execution == #
def main():
    explorer = Nutrition5kExplorer()
    explorer.generate_summary_report()
    
    # Optionally plot distributions
    if explorer.data is not None:
        try:
            print()
            response = input("Generate distribution plots? (y/n): ")
            if response.lower() == 'y':
                explorer.plot_distributions()
        except KeyboardInterrupt:
            print("\nSkipping plots.")
    else:
        print("\n⚠️  Cannot generate plots - no data loaded.")


if __name__ == "__main__":
    main()