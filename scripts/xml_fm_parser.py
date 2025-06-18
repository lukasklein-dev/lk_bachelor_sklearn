"""
%%%%% VaRA Feature Model XML Parser %%%%%
This script transforms the xml format used by FeatureIDE into the xml format used by vara-feature.
Comment: I first used the BFS approach for parsing the xml tree. But then I changed it into DFS, since this makes more sense for the parsing order.

Usage (How to):
Step 1: Create a Feature Model using FeatureIDE. NOTE: Until now, the script only supports binary features, mandatory/optional, and-groups, alternative-groups. So do not make use of or-groups or boolean constraints. Just construct all features as binary or using one hot encoding.
Step 2: TODO: In the script (code); enter the name of the feature model. (You might need to adjust input and output file paths.)
Step 3: Execute the script: (inside the root directory of the repo)
python scripts/xml_fm_parser.py --fm_name [xml_feature_model_name_without_extension]
-> Now the xml file should be parsed correctly and is ready to be used as input for vara-feature.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import click
import xml.etree.ElementTree as ET
from xml.dom import minidom

def parse_feature_model(node, parent=None, processed=None):
    """
    Recursive parser that traverses the feature model in a DFS manner.
    Args:
        node: Current XML node.
        parent: Parent feature name.
        processed: Set to track already processed feature names.
    """
    features = []

    # Initialize processed set if not provided
    if processed is None:
        processed = set()

    # Process only relevant nodes: <and>, <alt>, <feature>
    if node.tag in ["and", "alt", "feature"]:
        is_alt = node.tag == "alt"
        is_and = node.tag == "and"

        # Add the current node if it's an unprocessed feature
        if node.tag == "feature" and node.attrib["name"] not in processed:
            if "name" not in node.attrib:
                raise ValueError(f"Feature node is missing 'name' attribute: {ET.tostring(node, encoding='unicode')}")
            feature = {
                "name": node.attrib["name"],
                "optional": "mandatory" not in node.attrib or node.attrib["mandatory"] == "false",
                "parent": parent,
                "excluded": [],
            }
            features.append(feature)
            processed.add(node.attrib["name"])

        # Handle alternative (xor) or and-groups
        elif is_alt or is_and:
            current_name = node.attrib.get("name")
            for child in node: # "search deeper"
                child_name = child.attrib["name"]
                if child_name not in processed:
                    excluded = [
                        sibling.attrib["name"]
                        for sibling in node # search in the same level
                        if sibling.attrib["name"] != child_name
                    ] if is_alt else []
                    feature = {
                        "name": child_name,
                        "optional": False if is_alt else ("mandatory" not in child.attrib or child.attrib["mandatory"] == "false"),
                        "parent": current_name if is_and else parent,
                        "excluded": excluded,
                    }
                    features.append(feature)
                    processed.add(child_name)
                
                # Recur into child nodes
                features.extend(parse_feature_model(child, parent=child_name, processed=processed))
    
    return features


def generate_target_xml(features, vm_name):
    """Generate target XML format based on parsed features."""
    root = ET.Element("vm", name=vm_name)
    binary_options = ET.SubElement(root, "binaryOptions")

    for feature in features:
        config_option = ET.SubElement(binary_options, "configurationOption")
        ET.SubElement(config_option, "name").text = feature["name"]
        if feature["name"] == vm_name: # Root node has empty outputString
            ET.SubElement(config_option, "outputString")
        else:
            ET.SubElement(config_option, "outputString").text = feature["name"]
        ET.SubElement(config_option, "prefix")
        ET.SubElement(config_option, "postfix")
        ET.SubElement(config_option, "parent").text = feature["parent"] or ""
        ET.SubElement(config_option, "children")
        ET.SubElement(config_option, "impliedOptions")

        # Add excluded options
        excluded_options = ET.SubElement(config_option, "excludedOptions")
        for excluded in feature["excluded"]:
            ET.SubElement(excluded_options, "options").text = excluded

        ET.SubElement(config_option, "optional").text = str(feature["optional"]).capitalize()

    # Add empty numericOptions, booleanConstraints, and nonBooleanConstraints
    ET.SubElement(root, "numericOptions")
    ET.SubElement(root, "booleanConstraints")
    ET.SubElement(root, "nonBooleanConstraints")

    return root


def convert(input_file, output_file, vm_name):
    """Main function to parse XML and output XML."""
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Locate the <struct> element and parse its children
    struct_node = root.find("struct")
    if struct_node is None:
        raise ValueError("The <struct> element is missing in the input XML.")
    
    # Start parsing from the children of <struct>, skipping the container node itself
    features = []

    root_feature = {
        "name": vm_name,
        "optional": False,
        "parent": None,
        "excluded": [],
    }
    features.append(root_feature)

    for child in struct_node:
        features.extend(parse_feature_model(child))

    if not features:
        raise ValueError("No features were parsed. Check the input XML structure.")
    
    #print(f"Parsed features: {features}")  # LOGGING

    # Generate the target XML
    target_root = generate_target_xml(features, vm_name)

    # Prettify the XML using xml.dom.minidom
    rough_string = ET.tostring(target_root, encoding="utf-8", xml_declaration=False)
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Remove the first line (<?xml version="1.0" ?>)
    pretty_xml_without_declaration = "\n".join(pretty_xml.splitlines()[1:])

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty_xml_without_declaration.strip())

    print(f"{vm_name}.xml transformed and written to {output_file}") # LOGGING


def preprocessing_xml(input, output):
    # Filtering the node tag "graphics" and the parent name of the first and-group.
    first_alt = True
    try:
        with open(input, 'r', encoding='utf-8') as infile, open(output, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if "graphics" not in line.lower():
                    if ("<alt abstract=\"true\" mandatory=\"true\"" in line.lower()) & first_alt:
                        line = "		<alt abstract=\"true\" mandatory=\"true\" name=\"\">" + "\n"
                        first_alt = False
                    outfile.write(line)
    except FileNotFoundError:
        print(f"Error: File {input} not found.")
    except Exception as e:
        print(f"And error occurred: {e}")


@click.command()
@click.option('--fm_name', required=True, help='XML file name (without the .xml ending) of the feature model, e.g. bayesian_gaussian_mixture')
def main(fm_name):
    # CONFIGURATION
    input_xml = "scripts/io/mnist/feature_models/fm/" + fm_name + ".xml"
    input_pre_xml = "scripts/io/mnist/feature_models/fm_pre/" + fm_name + "-pre.xml"
    output_xml = "scripts/io/mnist/feature_models/fm_new/" + fm_name + ".xml"
    vm_name = fm_name # Don't change this name! (It is used as the root node name in the output XML)

    preprocessing_xml(input_xml, input_pre_xml) # Skip lines/tags other than and, alt, feature (i.e. skip graphics):

    convert(input_pre_xml, output_xml, vm_name) # Start Skript

if __name__ == "__main__":
    main()