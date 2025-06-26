import ast

def insert_config_ids(case_study_path, output_path):
    with open(case_study_path, "r") as f:
        lines = f.readlines()

    # Find config_type section and collect all config ids
    config_ids = []
    in_config_section = False
    config_lines = {}
    for idx, line in enumerate(lines):
        if line.strip() == "config_type: PlainCommandlineConfiguration":
            in_config_section = True
            continue
        if in_config_section:
            if line.strip() == "" or line.startswith("---"):
                break
            # Get id at the start of the line (before ':')
            parts = line.strip().split(":", 1)
            if parts and parts[0].isdigit():
                cid = int(parts[0])
                config_ids.append(cid)
                config_lines[idx] = (cid, line)

    # Find the line with "config_ids:" and insert "- X" for each id after it
    new_lines = []
    i = 0
    while i < len(lines):
        # If this is a config line, modify it to add the id as first element in the feature list
        if i in config_lines:
            cid, orig_line = config_lines[i]
            # Extract the list string
            quote_idx = orig_line.find("'")
            if quote_idx != -1:
                list_str = orig_line[quote_idx+1:orig_line.rfind("'")]
                features = ast.literal_eval(list_str)
                # Insert the config_id as first element (as string) if not already present
                if not features or str(cid) != str(features[0]):
                    features = [str(cid)] + features
                # Rebuild the line
                new_list_str = str(features).replace("'", '"')
                new_line = f"{cid}:".ljust(17) + f"'{new_list_str}'\n"
                new_lines.append(new_line)
            else:
                new_lines.append(orig_line)
        else:
            new_lines.append(lines[i])
        # Insert config_ids after the "config_ids:" line
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
    with open(output_path, "w") as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    # TODO: Change in and out paths to your case study file:
    input="/home/lukas/Schreibtisch/repos/lk-bachelor-sklearn/scripts/io/mnist/configurations/cs_files/in/sklearn_experiment_mnb_10.case_study"
    output="/home/lukas/Schreibtisch/repos/lk-bachelor-sklearn/scripts/io/mnist/configurations/cs_files/out/sklearn_experiment_mnb_10.case_study"
    insert_config_ids(input, output)