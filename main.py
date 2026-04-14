# ============================================================
# Student Performance - AI Data Analyzer
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -- SECTION 1: LOAD DATA --
def load_data():
    """Loads the student dataset from CSV file."""
    data = pd.read_csv("data/student_data.csv")
    print("Dataset loaded successfully!")
    print(f"   -> {data.shape[0]} students, {data.shape[1]} columns\n")
    return data

# -- SECTION 2: CLEAN DATA --
def clean_data(data):
    """Checks for missing values and removes duplicates."""
    print("Checking for missing values...")
    if data.isnull().sum().sum() == 0:
        print("   -> No missing values found!\n")
    else:
        data.dropna(inplace=True)
        print("   -> Missing values dropped.\n")
    data.drop_duplicates(inplace=True)
    return data

# -- SECTION 3: ANALYZE DATA --
def analyze_data(data):
    """Performs statistical analysis using Pandas and NumPy."""
    print("=" * 50)
    print("STATISTICAL ANALYSIS")
    print("=" * 50)

    print(f"\nTotal Students        : {len(data)}")
    print(f"Average Age           : {np.mean(data['age']):.1f} years")
    print(f"Average G1 (1st term) : {np.mean(data['G1']):.2f}/20")
    print(f"Average G2 (2nd term) : {np.mean(data['G2']):.2f}/20")
    print(f"Average G3 (final)    : {np.mean(data['G3']):.2f}/20")
    print(f"Highest Final Grade   : {data['G3'].max()}/20")
    print(f"Lowest Final Grade    : {data['G3'].min()}/20")

    passed = (data['G3'] >= 10).sum()
    failed = (data['G3'] < 10).sum()
    print(f"\nStudents who Passed   : {passed} ({passed/len(data)*100:.1f}%)")
    print(f"Students who Failed   : {failed} ({failed/len(data)*100:.1f}%)\n")

    corr = np.corrcoef(data['G1'], data['G3'])[0, 1]
    print(f"Correlation G1 vs G3  : {corr:.2f}")
    print("   -> Strong correlation: students who start well tend to finish well.\n")

# -- SECTION 4: VISUALIZE DATA --
def visualize_data(data):
    """Creates 3 visualizations from the real dataset."""
    print("Generating visualizations...")
    sns.set_theme(style="darkgrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Student Performance - Data Analysis", fontsize=15, fontweight='bold')

    # Chart 1: Age Distribution
    axes[0].hist(data['age'], bins=8, color='steelblue', edgecolor='white')
    axes[0].axvline(data['age'].mean(), color='red', linestyle='--',
                    label=f"Mean: {data['age'].mean():.1f}")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Number of Students")
    axes[0].legend()

    # Chart 2: Final Grade (G3) Distribution
    axes[1].hist(data['G3'], bins=10, color='mediumseagreen', edgecolor='white')
    axes[1].axvline(data['G3'].mean(), color='red', linestyle='--',
                    label=f"Mean: {data['G3'].mean():.1f}")
    axes[1].set_title("Final Grade (G3) Distribution")
    axes[1].set_xlabel("Grade (0-20)")
    axes[1].set_ylabel("Number of Students")
    axes[1].legend()

    # Chart 3: G1 vs G3 Scatter
    axes[2].scatter(data['G1'], data['G3'], alpha=0.5, color='coral', edgecolors='white')
    axes[2].set_title("G1 vs G3 - Does 1st Term Predict Final Grade?")
    axes[2].set_xlabel("G1 (1st Term Grade)")
    axes[2].set_ylabel("G3 (Final Grade)")

    plt.tight_layout()
    plt.savefig("student_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved as 'student_analysis.png'\n")

# -- MAIN --
def main():
    print("\n" + "=" * 50)
    print("  STUDENT PERFORMANCE - AI DATA ANALYZER")
    print("=" * 50 + "\n")
    data = load_data()
    data = clean_data(data)
    analyze_data(data)
    visualize_data(data)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
    