from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.app import AppInterface
from src.utils import plot_target_distribution, plot_correlation_heatmap


def main():
    print("Startar main()...")

    processor = DataProcessor(path="data/heart.csv")
    df = processor.load()
    df = processor.clean(df)

    # EDA (sparar figurer i outputs/)
    plot_target_distribution(df, target_col="target", output_dir="outputs")
    plot_correlation_heatmap(df, output_dir="outputs")

    X, y = processor.split_xy(df, target_col="target")

    trainer = ModelTrainer(model_type="logreg")  # byt till "rf" om du vill
    model, X_test, y_test = trainer.train(X, y, test_size=0.2)
    results = trainer.evaluate(model, X_test, y_test)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print("Confusion matrix:\n", results["confusion_matrix"])
    print("\nClassification report:\n", results["classification_report"])
    if "roc_auc" in results:
        print(f"ROC AUC: {results['roc_auc']:.3f}")

    trainer.save(model, filepath="model.joblib")
    print("\nModel saved to model.joblib")

    app = AppInterface(model=model, feature_names=list(X.columns))
    app.run()


if __name__ == "__main__":
    main()
