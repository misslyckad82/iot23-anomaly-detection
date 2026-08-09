import pandas as pd

# Sökvägen till din conn.log.labeled-fil
input_file = r"C:\Users\missc\Downloads\iot_23_datasets_full\opt\Malware-Project\BigDataset\IoTScenarios\CTU-IoT-Malware-Capture-9-1\bro\conn.log.labeled"

# Där CSV-filen ska sparas
output_file = r"C:\Users\missc\Desktop\AnomalyDetectionProject\output\iot23.csv"

# Läs filen
with open(input_file, "r") as f:
    lines = f.readlines()

fields = None
data = []

for line in lines:
    line = line.strip()

    # Hämta kolumnnamnen
    if line.startswith("#fields"):
        fields = line.split("\t")[1:]

    # Hoppa över kommentarer
    elif line.startswith("#"):
        continue

    # Data
    else:
        data.append(line.split("\t"))

# Skapa DataFrame
df = pd.DataFrame(data, columns=fields)

# Spara som CSV
df.to_csv(output_file, index=False)

print(f"Klar! CSV sparad som {output_file}")