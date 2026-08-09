import pandas as pd

# Sökvägen till den stora CSV-filen
input_file = r"C:\Users\missc\Desktop\AnomalyDetectionProject\output\iot23.csv"

# Den mindre filen som ska skapas
output_file = r"C:\Users\missc\Desktop\AnomalyDetectionProject\output\iot23_small.csv"

# Hur många rader som ska tas med
rows = 400000

print("Läser CSV...")
df = pd.read_csv(input_file, nrows=rows)

print("Sparar mindre CSV...")
df.to_csv(output_file, index=False)

print(f"Klar! Sparade {len(df)} rader till:")
print(output_file)