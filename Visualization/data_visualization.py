import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os


class DataVisualizer:
    """
    A flexible class for visualizing CSV data without modifying it.
    Can adapt to different CSV structures with varying columns.
    """
    
    def __init__(self, csv_path):
        """Initialize with path to CSV file"""
        self.csv_path = csv_path
        self.data = None
        self._load_data()
        
    def _load_data(self):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Successfully loaded data with shape: {self.data.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def get_summary_statistics(self):
        """Display basic summary statistics"""
        if self.data is None:
            return
        
        print("\n==== Summary Statistics ====")
        print(self.data.describe().T)
        
        # Count missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("\n==== Missing Values ====")
            print(missing_values[missing_values > 0])
        
    def plot_distributions(self, sample_size=None):
        """Plot distribution of each numerical column"""
        if self.data is None:
            return
            
        numerical_cols = self.data.select_dtypes(include=np.number).columns
        
        if len(numerical_cols) == 0:
            print("No numerical columns to plot distributions for.")
            return
            
        # Sample data if specified
        plot_data = self.data.sample(sample_size) if sample_size else self.data
        
        n_cols = min(len(numerical_cols), 3)
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                sns.histplot(plot_data[col], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for j in range(len(numerical_cols), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = Path('visualization_output')
        output_dir.mkdir(exist_ok=True)
        
        plt.savefig(output_dir / 'distributions.png')
        plt.close()
        print(f"Saved distributions plot to {output_dir / 'distributions.png'}")
        
    def plot_boxplots(self):
        """Create boxplots to identify outliers"""
        if self.data is None:
            return
            
        numerical_cols = self.data.select_dtypes(include=np.number).columns
        
        if len(numerical_cols) == 0:
            print("No numerical columns to plot boxplots for.")
            return
            
        n_cols = min(len(numerical_cols), 3)
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                sns.boxplot(y=self.data[col], ax=axes[i])
                axes[i].set_title(f'Boxplot of {col}')
        
        # Hide unused subplots
        for j in range(len(numerical_cols), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = Path('visualization_output')
        output_dir.mkdir(exist_ok=True)
        
        plt.savefig(output_dir / 'boxplots.png')
        plt.close()
        print(f"Saved boxplots to {output_dir / 'boxplots.png'}")
        
    def plot_correlation_matrix(self):
        """Plot correlation matrix for numerical columns"""
        if self.data is None:
            return
            
        numerical_data = self.data.select_dtypes(include=np.number)
        
        if numerical_data.shape[1] < 2:
            print("Need at least 2 numerical columns to plot correlation matrix.")
            return
            
        plt.figure(figsize=(12, 10))
        corr = numerical_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
        plt.title('Correlation Matrix')
        
        # Create output directory if it doesn't exist
        output_dir = Path('visualization_output')
        output_dir.mkdir(exist_ok=True)
        
        plt.savefig(output_dir / 'correlation_matrix.png')
        plt.close()
        print(f"Saved correlation matrix to {output_dir / 'correlation_matrix.png'}")
        
    def plot_time_series(self, time_col=None):
        """
        Plot time series data if available.
        If time_col is None, will try to automatically detect a time column or use the index.
        """
        if self.data is None:
            return
        
        numerical_cols = self.data.select_dtypes(include=np.number).columns
        
        # Try to identify time column if not specified
        if time_col is None:
            # Check for common time column names
            time_column_names = ['time', 'timestamp', 'date', 'datetime']
            for col in time_column_names:
                if col in self.data.columns:
                    time_col = col
                    break
        
        # Use index as x-axis if no time column found
        x_axis = self.data[time_col] if time_col else self.data.index
        
        # Plot each numerical column as a time series
        numerical_cols = [col for col in numerical_cols if col != time_col]
        
        if len(numerical_cols) == 0:
            print("No numerical columns to plot as time series.")
            return
        
        # Plot at most 9 series per figure to avoid overcrowding
        max_plots_per_figure = 9
        for i in range(0, len(numerical_cols), max_plots_per_figure):
            subset_cols = numerical_cols[i:i+max_plots_per_figure]
            n_cols = min(3, len(subset_cols))
            n_rows = (len(subset_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            
            for j, col in enumerate(subset_cols):
                if j < len(axes):
                    axes[j].plot(x_axis, self.data[col])
                    axes[j].set_title(f'Time Series: {col}')
                    axes[j].set_xlabel('Time' if time_col else 'Index')
                    axes[j].set_ylabel(col)
            
            # Hide unused subplots
            for k in range(len(subset_cols), len(axes)):
                axes[k].set_visible(False)
                
            plt.tight_layout()
            
            # Create output directory if it doesn't exist
            output_dir = Path('visualization_output')
            output_dir.mkdir(exist_ok=True)
            
            plt.savefig(output_dir / f'time_series_{i//max_plots_per_figure}.png')
            plt.close()
            print(f"Saved time series plot to {output_dir / f'time_series_{i//max_plots_per_figure}.png'}")
            
    def plot_pairplot(self, sample_size=1000, columns=None):
        """
        Create a pairplot for selected numerical columns.
        
        Args:
            sample_size: Number of samples to use (to avoid overwhelming plots)
            columns: List of columns to include. If None, uses all numerical columns 
                     (up to a maximum of 5 to avoid excessive plots)
        """
        if self.data is None:
            return
            
        numerical_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        
        if len(numerical_cols) < 2:
            print("Need at least 2 numerical columns to create pairplot.")
            return
        
        # If columns not specified, use all numerical columns (up to 5)
        if columns is None:
            columns = numerical_cols[:min(5, len(numerical_cols))]
        
        # Sample data to avoid overwhelming plots
        plot_data = self.data.sample(min(sample_size, len(self.data)))
        
        plt.figure(figsize=(12, 10))
        sns.pairplot(plot_data[columns])
        
        # Create output directory if it doesn't exist
        output_dir = Path('visualization_output')
        output_dir.mkdir(exist_ok=True)
        
        plt.savefig(output_dir / 'pairplot.png')
        plt.close()
        print(f"Saved pairplot to {output_dir / 'pairplot.png'}")
    
    def visualize_all(self, sample_size=1000):
        """Run all visualization methods"""
        self.get_summary_statistics()
        self.plot_distributions(sample_size)
        self.plot_boxplots()
        self.plot_correlation_matrix()
        self.plot_time_series()
        self.plot_pairplot(sample_size)
        
        print("\nAll visualizations complete! Check the 'visualization_output' directory for the generated plots.")


def main():
    """Main function to run the visualization pipeline"""
    # Update file path to point to Data/test_x.csv
    file_path = 'Data/test_x.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Please specify a valid CSV file path.")
        file_path = input("Enter CSV file path: ")
    
    # Create visualizer and run all visualizations
    visualizer = DataVisualizer(file_path)
    visualizer.visualize_all()


if __name__ == "__main__":
    main()
