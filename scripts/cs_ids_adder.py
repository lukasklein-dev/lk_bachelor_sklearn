def insert_config_ids(case_study_path):
    with open(case_study_path, "r") as f:
        lines = f.readlines()

    # Find config_type section and collect all config ids
    config_ids = []
    in_config_section = False
    for line in lines:
        if line.strip() == "config_type: PlainCommandlineConfiguration":
            in_config_section = True
            continue
        if in_config_section:
            if line.strip() == "" or line.startswith("---"):
                break
            # Get id at the start of the line (before ':')
            parts = line.strip().split(":")
            if parts and parts[0].isdigit():
                config_ids.append(int(parts[0]))

    # Find the line with "config_ids:" and insert "- X" for each id after it
    new_lines = []
    i = 0
    while i < len(lines):
        new_lines.append(lines[i])
        if lines[i].strip() == "config_ids:":
            for cid in config_ids:
                new_lines.append(f"    - {cid}\n")
            # Skip any existing "- X" lines
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("- "):
                j += 1
            i = j - 1
        i += 1

    # Write back to file (or print for testing)
    with open(case_study_path, "w") as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    insert_config_ids("scripts/io/mnist/configurations/sklearn_experiment_0.case_study")