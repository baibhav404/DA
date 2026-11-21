
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Project folder location
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Titanic-Dataset.csv"
OUTPUT_DIR = BASE_DIR / "titanic_outputs"


def load_data():
    """Load Titanic passenger information."""
    if not DATA_PATH.exists():
        print("Error: Titanic-Dataset.csv not found!")
        return None

    df = pd.read_csv(DATA_PATH)

    print("\n===== TITANIC DATASET =====")
    print(df.head())
    print("\nDataset Shape:", df.shape)

    return df


def clean_data(df):
    """Handle missing values and create new columns."""
    df = df.copy()

    # Fill missing age values using class-wise median
    age_values = df.groupby("Pclass")["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_values)

    # Fill missing embarkation values
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

    # Create family-related columns
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Convert cabin information into a binary feature
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Categorize passengers by age
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
        include_lowest=True
    )

    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    return df


def analyze_data(df):
    """Display important survival statistics."""

    print("\n===== SURVIVAL ANALYSIS =====")

    print("\nOverall Survival Rate:")
    print(round(df["Survived"].mean() * 100, 2), "%")

    print("\nSurvival by Gender:")
    print(
        (df.groupby("Sex")["Survived"].mean() * 100).round(2)
    )

    print("\nSurvival by Passenger Class:")
    print(
        (df.groupby("Pclass")["Survived"].mean() * 100).round(2)
    )

    print("\nSurvival by Age Group:")
    print(
        (df.groupby("AgeGroup", observed=False)["Survived"].mean() * 100).round(2)
    )

    print("\nAverage Ticket Fare:")
    print(df.groupby("Pclass")["Fare"].mean().round(2))

    print("\nSurvival by Family Size:")
    print(
        (df.groupby("FamilySize")["Survived"].mean() * 100).round(2)
    )


def visualize_data(df):
    """Generate charts for passenger survival analysis."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Chart 1: Survival Count
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Survived", hue="Survived",
                  palette="Set2", legend=False)
    plt.title("Titanic Passenger Survival Count")
    plt.xlabel("Survival Status (0 = No, 1 = Yes)")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_count.png")
    plt.close()

    # Chart 2: Gender Comparison
    gender_data = df.groupby("Sex")["Survived"].mean().reset_index()
    gender_data["Survived"] *= 100

    plt.figure(figsize=(7, 5))
    sns.barplot(data=gender_data, x="Sex", y="Survived",
                hue="Sex", palette="pastel", legend=False)
    plt.title("Survival Rate by Gender")
    plt.ylabel("Survival Percentage")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gender_survival.png")
    plt.close()

    # Chart 3: Passenger Class
    class_data = df.groupby("Pclass")["Survived"].mean().reset_index()
    class_data["Survived"] *= 100

    plt.figure(figsize=(7, 5))
    sns.barplot(data=class_data, x="Pclass", y="Survived",
                hue="Pclass", palette="viridis", legend=False)
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Percentage")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_survival.png")
    plt.close()

    # Chart 4: Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Age", hue="Survived",
                 bins=25, kde=True, palette="Set1")
    plt.title("Age Distribution of Passengers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_distribution.png")
    plt.close()

    # Chart 5: Fare Analysis
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Pclass", y="Fare",
                hue="Pclass", palette="Set3", legend=False)
    plt.title("Ticket Fare Comparison")
    plt.xlabel("Passenger Class")
    plt.ylabel("Ticket Fare")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fare_comparison.png")
    plt.close()

    # Chart 6: Correlation Heatmap
    plt.figure(figsize=(9, 6))
    numeric_df = df.select_dtypes(include="number")

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()

    print("\nAll charts saved successfully!")


def save_results(df):
    """Export cleaned data and analysis reports."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    df.to_csv(
        OUTPUT_DIR / "cleaned_titanic_data.csv",
        index=False
    )

    gender_report = df.groupby("Sex")["Survived"].mean() * 100
    gender_report.to_csv(OUTPUT_DIR / "gender_report.csv")

    class_report = df.groupby("Pclass")["Survived"].mean() * 100
    class_report.to_csv(OUTPUT_DIR / "class_report.csv")

    print("Analysis reports exported successfully!")


def show_conclusion(df):
    """Print the major findings of the project."""

    female = df[df["Sex"] == "female"]["Survived"].mean() * 100
    male = df[df["Sex"] == "male"]["Survived"].mean() * 100

    first_class = df[df["Pclass"] == 1]["Survived"].mean() * 100
    third_class = df[df["Pclass"] == 3]["Survived"].mean() * 100

    print("\n========== PROJECT CONCLUSION ==========")

    print(f"Female Survival Rate: {female:.2f}%")
    print(f"Male Survival Rate: {male:.2f}%")

    print(f"First Class Survival: {first_class:.2f}%")
    print(f"Third Class Survival: {third_class:.2f}%")

    print("\nKey Observation:")
    print("Gender and passenger class show clear")
    print("associations with passenger survival.")

    print("\nProject Completed Successfully!")


def main():
    passengers = load_data()

    if passengers is None:
        return

    passengers = clean_data(passengers)

    analyze_data(passengers)
    visualize_data(passengers)
    save_results(passengers)
    show_conclusion(passengers)


if __name__ == "__main__":
    main()
