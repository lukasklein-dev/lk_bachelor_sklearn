import re

input_path = "scripts/io/mnist/configurations/cs_files/out/sklearn_experiment_mnb_10.case_study"
output_path = "scripts/io/mnist/configurations/cs_files/cleaned/sklearn_experiment_mnb_10.case_study"

with open(input_path, "r") as f:
    text = f.read()

# Entfernt '"n_jobs", "-1"' (mit oder ohne Komma davor/dahinter, inkl. Leerzeichen)
text = re.sub(r',?\s*"n_jobs",\s*"-1"', '', text)
text = re.sub(r'"n_jobs",\s*"-1"\s*,?', '', text)

# Entfernt '"random_state", "0"' (mit oder ohne Komma davor/dahinter, inkl. Leerzeichen)
text = re.sub(r',?\s*"random_state",\s*"0"', '', text)
text = re.sub(r'"random_state",\s*"0"\s*,?', '', text)

# Optional: Doppelte Kommas und überflüssige Leerzeichen nach dem Entfernen bereinigen
text = re.sub(r',\s*,', ',', text)
text = re.sub(r'\[\s*,', '[', text)
text = re.sub(r',\s*\]', ']', text)

with open(output_path, "w") as f:
    f.write(text)