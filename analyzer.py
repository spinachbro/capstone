import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class Analyzer:
    def __init__(self):
        self.data = None

    def read_dataset(self, file_name):
        """
        A function that takes the name of a CSV file, reads the file, and saves the data in the instance of the analyzer class.
        """
        # Read the CSV file
        self.data = pd.read_csv(file_name)

        # Display basic information about the dataset
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")

    def describe(self):
        """
        Prints the input features' attribute types and shows basic statistical analysis on the numeric data attributes. Retrieves the data of all the books in the database.
        """
        # Display data types
        print("Data Types:")
        print(self.data.dtypes)
        print("\n" + "="*50 + "\n")

        # Display basic info
        print("Dataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print("\n" + "="*50 + "\n")

        # # Statistical description for numeric columns
        # print("Statistical Description:")
        # print(self.data.describe())
        # print("\n" + "="*50 + "\n")

        # Check for missing values
        print("Missing Values:")
        print(self.data.isnull().sum())

    def drop_missing_data(self):
        """
        Drops any data sample with missing values.
        """
        # Check if there are any missing values
        missing_before = self.data.isnull().sum().sum()
        rows_before = len(self.data)

        # Drop rows with any missing values
        self.data = self.data.dropna()

        # Display results
        rows_after = len(self.data)
        missing_after = self.data.isnull().sum().sum()

        print(f"Missing values before: {missing_before}")
        print(f"Rows before: {rows_before}")
        print(f"Rows after: {rows_after}")
        print(f"Rows dropped: {rows_before - rows_after}")
        print(f"Missing values after: {missing_after}")

    def drop_columns(self, columns_to_drop):
        """
        A function that takes a list of attribute names to drop from the dataset.
        """
        # Check if all columns exist before dropping
        existing_columns = []
        missing_columns = []

        for col in columns_to_drop:
            if col in self.data.columns:
                existing_columns.append(col)
            else:
                missing_columns.append(col)

        # Drop existing columns
        if existing_columns:
            self.data = self.data.drop(columns=existing_columns)
            print(f"Dropped columns: {existing_columns}")

        # Report missing columns
        if missing_columns:
            print(f"Warning: These columns were not found: {missing_columns}")
        
        print(f"Remaining columns: {list(self.data.columns)}")
        print(f"New shape: {self.data.shape}")

    def encode_features(self, nominal_columns):
        """
        A function that takes a list of nominal column names and encodes the values of these columns in the dataset.
        """
      # Initialize label encoders dictionary to store encoders for each column
        self.label_encoders = {}
        print("Encoding nominal features...")
        for column in nominal_columns:
            if column in self.data.columns:
                # Create label encoder for this column
                le = LabelEncoder()
                # Fit and transform the column
                self.data[column] = le.fit_transform(self.data[column])
                # Store the encoder for potential inverse transformation later
                self.label_encoders[column] = le
                print(f"Encoded column '{column}': {len(le.classes_)} unique values")
                print(f"  Classes: {list(le.classes_)}")
            else:
                print(f"Warning: Column '{column}' not found in dataset")

        print(f"Encoding complete. Encoded {len(self.label_encoders)} columns.")

    def encode_label(self, target_column):
        """
        A function that takes the target name to encode in the dataset for classification tasks.
        """

        if target_column not in self.data.columns:
            print(f"Error: Column '{target_column}' not found in dataset")
            return

        # Create label encoder for target
        self.target_encoder = LabelEncoder()

        # Store original target values before encoding
        original_values = self.data[target_column].unique()

        # Fit and transform the target column
        self.data[target_column] = self.target_encoder.fit_transform(self.data[target_column])

        print(f"Encoded target column '{target_column}'")
        print(f"Original classes: {sorted(original_values)}")
        print(f"Encoded classes: {sorted(self.data[target_column].unique())}")
        print(f"Number of classes: {len(self.target_encoder.classes_)}")

    def shuffle(self):
        """
        A function that shuffles the data samples.
        """
        # Store original shape for confirmation
        original_shape = self.data.shape

        # Set random seed for reproducibility
        np.random.seed(42)

        # Shuffle the dataframe
        self.data = self.data.sample(frac=1,random_state=42).reset_index(drop=True)

        print("Data shuffled successfully!")
        print(f"Shape remains: {self.data.shape}")
        print("Index has been reset.")
        print("First few rows after shuffle:")
        print(self.data.head())
    
    def retrieve_data(self):
        """
        A function that simply returns the dataset stored in an analyzer instance.
        """
        if self.data is None:
            print("Warning: No data has been loaded yet.")
            return None

        print(f"Retrieving dataset with shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")

        return self.data
    
    def plot_correlationMatrix(self):
        """
        A function that plots an annotated correlation matrix and saves the plot with the title, "Correlation.png."
        """
        # Get numerical columns only
        numerical_data = self.data.select_dtypes(include=['int64', 'float64'])

        if len(numerical_data.columns) < 2:
            print("Error: Not enough numerical columns for correlation analysis")
            return

        # Calculate correlation matrix
        correlation_matrix = numerical_data.corr()

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Create heatmap with annotations
        sns.heatmap(correlation_matrix,
                    annot=True,           # Show correlation values
                    cmap='coolwarm',      # Color scheme
                    center=0,             # Center colormap at 0
                    square=True,          # Make cells square
                    fmt='.2f',            # Format numbers to 2 decimal places
                    cbar_kws={'shrink': 0.8})

        plt.title('Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()

        # Save the plot
        plt.savefig('Correlation.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print("Correlation matrix plotted and saved as 'Correlation.png'")
        print(f"Matrix includes {len(numerical_data.columns)} numerical columns:")
        print(f"Columns: {list(numerical_data.columns)}")

    def plot_pairPlot(self):
        """
        A function that plots the pair-plot between attributes of the input dataset.
        """
        # Get numerical columns only for pair plot
        numerical_data = self.data.select_dtypes(include=['int64',
    'float64'])

        if len(numerical_data.columns) < 2:
            print("Error: Not enough numerical columns for pair plot")
            return

        print("Creating pair plot...")
        print(f"Plotting relationships between {len(numerical_data.columns)} numerical columns:")
        print(f"Columns: {list(numerical_data.columns)}")

        # Create pair plot
        plt.figure(figsize=(12, 10))

        # Use seaborn's pairplot function
        pair_plot = sns.pairplot(numerical_data,
                                diag_kind='hist', # Histograms on diagonal
                                plot_kws={'alpha': 0.6, 's': 20},  # Scatter plot settings
                                diag_kws={'bins': 30}) # Histogram settings

        # Adjust layout
        pair_plot.fig.suptitle('Pair Plot of Numerical Features', y=1.02, fontsize=16)
        plt.tight_layout()

        # Save the plot
        plt.savefig('PairPlot.png', dpi=300,bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print("Pair plot created and saved as 'PairPlot.png'")
        print("The plot shows:")
        print("- Diagonal: Distribution of each feature")
        print("- Off-diagonal: Scatter plots between feature pairs")


    def plot_histograms_numeric(self):
        """
        A function that plots the histogram of the continuous numerical attributes in the input dataset.
        """
        # Get numerical columns only
        numerical_data = self.data.select_dtypes(include=['int64',
    'float64'])

        if len(numerical_data.columns) == 0:
            print("Error: No numerical columns found for histogram plotting")
            return

        print("Creating histograms for numerical features...")
        print(f"Plotting histograms for {len(numerical_data.columns)} numerical columns:")
        print(f"Columns: {list(numerical_data.columns)}")

        # Calculate subplot layout
        n_cols = min(3, len(numerical_data.columns))  # Max 3 columns
        n_rows = (len(numerical_data.columns) + n_cols -1) // n_cols  # Ceiling division
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols,figsize=(15, 5 * n_rows))

        # Handle case where there's only one subplot
        if len(numerical_data.columns) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()

        # Plot histogram for each numerical column
        for i, column in enumerate(numerical_data.columns):
            axes[i].hist(numerical_data[column], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {column}', fontsize=12)
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)

        # Hide any extra subplots
        for i in range(len(numerical_data.columns), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Histograms of Numerical Features', fontsize=16, y=0.98)
        plt.tight_layout()

        # Save the plot
        plt.savefig('Histograms_Numeric.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print("Histograms plotted and saved as 'Histograms_Numeric.png'")
        print("Each histogram shows the distribution of values for numerical features")

    def plot_histograms_categorical(self):
        """
        A function that plots the histogram of the nominal attributes in the input dataset.
        """
        # Use encoded categorical columns
        if not hasattr(self, 'label_encoders') or not self.label_encoders:
            print("Error: No encoded categorical columns found")
            return

        categorical_columns = list(self.label_encoders.keys())
        print(f"Plotting histograms for: {categorical_columns}")

        # Create one subplot per column
        fig, axes = plt.subplots(1,len(categorical_columns), figsize=(5 *
    len(categorical_columns), 4))

        # Ensure axes is always iterable
        if len(categorical_columns) == 1:
            axes = [axes]

        # Plot each column
        for i, column in enumerate(categorical_columns):
            value_counts = self.data[column].value_counts().sort_index()
            axes[i].bar(range(len(value_counts)), value_counts.values)
            axes[i].set_title(f'{column}')
            axes[i].set_xticks(range(len(self.label_encoders[column].classes_)))
            axes[i].set_xticklabels(self.label_encoders[column].classes_, rotation=45)

        plt.tight_layout()
        plt.savefig('Histograms_Categorical.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_boxPlot(self):
        """
        A function that plots the box-plot of attributes of the input dataset.
        """
        numerical_data = self.data.select_dtypes(include=['int64',
    'float64'])

        if len(numerical_data.columns) == 0:
            print("Error: No numerical columns found for box plot")
            return

        print(f"Creating box plots for: {list(numerical_data.columns)}")

        # Separate price from other features
        other_cols = [col for col in numerical_data.columns if col != 'price']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Price box plot
        if 'price' in numerical_data.columns:
            ax1.boxplot([numerical_data['price'].dropna()], labels=['price'])
            ax1.set_title('Price Box Plot')
            ax1.grid(True, alpha=0.3)

        # Other features box plot
        if other_cols:
            ax2.boxplot([numerical_data[col].dropna() for col in other_cols], labels=other_cols)
            ax2.set_title('Other Features Box Plots')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('BoxPlot.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Box plots saved as 'BoxPlot.png'")

# Execution flow

analyzer = Analyzer()

print("*******************")
print("                   ")
print("                   ")
print("READING THE DATASET")
analyzer.read_dataset('diamonds.csv')

print("*******************")
print("                   ")
print("                   ")
print("DESCRIBING")
analyzer.describe()

print("*******************")
print("                   ")
print("                   ")
print("DROPPING MISSING")
analyzer.drop_missing_data()

print("*******************")
print("                   ")
print("                   ")
print("DROPPING COLUMNS")
analyzer.drop_columns(['Unnamed: 0'])

print("*******************")
print("                   ")
print("                   ")
print("ENCODING COLUMNS")
analyzer.encode_features(['cut', 'color', 'clarity'])

print("*******************")
print("                   ")
print("                   ")
print("SAMPLE DATA")
print(analyzer.data.head())

# print("*******************")
# print("                   ")
# print("                   ")
# print("ENCODING label")
# analyzer.encode_label()

print("*******************")
print("                   ")
print("                   ")
print("SHUFFLING DATA")
analyzer.shuffle()

print("*******************")
print("                   ")
print("                   ")
print("RETRIEVING DATA")
final_data = analyzer.retrieve_data()

# Plotting functions commented out to avoid repetitive output in ML tasks
# Uncomment these lines if you want to run exploratory data analysis separately

# print("*******************")
# print("                   ")
# print("                   ")
# print("PLOTTING CORRELATION MATRIX")
# analyzer.plot_correlationMatrix()

# print("*******************")
# print("                   ")
# print("                   ")
# print("PLOTTING PAIR PLOT")
# analyzer.plot_pairPlot()

# print("*******************")
# print("                   ")
# print("                   ")
# print("PLOTTING HISTOGRAMS")
# analyzer.plot_histograms_numeric()

# print("*******************")
# print("                   ")
# print("                   ")
# print("PLOTTING CATEGORICAL HISTOGRAMS")
# analyzer.plot_histograms_categorical()

# print("*******************")
# print("                   ")
# print("                   ")
# print("PLOTTING BOX PLOTS")
# analyzer.plot_boxPlot()