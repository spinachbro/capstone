#!/usr/bin/env python3
"""
Exploratory Data Analysis Script
Run this separately when you want to generate plots and analyze the data
"""

from analyzer import Analyzer

def run_eda():
    """Run comprehensive exploratory data analysis"""
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = Analyzer()
    
    # Data preparation
    analyzer.read_dataset('diamonds.csv')
    analyzer.drop_missing_data()
    analyzer.drop_columns(['Unnamed: 0'])
    analyzer.encode_features(['cut', 'color', 'clarity'])
    
    print("\n📊 GENERATING VISUALIZATIONS...")
    print("-" * 40)
    
    # Generate all plots
    print("\n🔗 Creating correlation matrix...")
    analyzer.plot_correlationMatrix()
    
    print("\n📈 Creating pair plot...")
    analyzer.plot_pairPlot()
    
    print("\n📊 Creating numeric histograms...")
    analyzer.plot_histograms_numeric()
    
    print("\n📊 Creating categorical histograms...")
    analyzer.plot_histograms_categorical()
    
    print("\n📦 Creating box plots...")
    analyzer.plot_boxPlot()
    
    print("\n✅ EDA COMPLETED!")
    print("📁 All plots saved as PNG files in current directory:")
    print("   - Correlation.png")
    print("   - PairPlot.png") 
    print("   - Histograms_Numeric.png")
    print("   - Histograms_Categorical.png")
    print("   - BoxPlot.png")

if __name__ == "__main__":
    run_eda()