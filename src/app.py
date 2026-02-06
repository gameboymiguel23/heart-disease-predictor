import numpy as np

class AppInterface:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def run(self):
        print("\n=== Heart Disease Predictor (Terminal) ===")
        print("Mata in värden för varje feature. Skriv 'q' för att avsluta.\n")

        while True:
            values = []
            for feat in self.feature_names:
                user_in = input(f"{feat}: ")
                if user_in.lower().strip() == "q":
                    print("Avslutar.")
                    return
                try:
                    values.append(float(user_in))
                except ValueError:
                    print("Ogiltigt värde. Börja om.\n")
                    values = None
                    break

            if values is None:
                continue

            x = np.array(values).reshape(1, -1)
            pred = self.model.predict(x)[0]

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(x)[0][1]
                print(f"\nPrediktion: {pred} (sannolikhet för sjukdom: {proba:.2f})\n")
            else:
                print(f"\nPrediktion: {pred}\n")
