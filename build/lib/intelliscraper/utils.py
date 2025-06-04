import json
import re

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    """ Remove spaces and special characters from text """
    return re.sub(r'\s+', '', text)

def element_to_string(element):
    """Convert an element to a string representation"""
    return f"{element.name} {' '.join([f'{k}={v}' for k, v in element.attrs.items()])}"


def get_most_similar_element(current_html, target_json):
    rule_json = json.loads(target_json)
    paths = parse_rules_to_paths(rule_json)
    soup = BeautifulSoup(current_html, 'html.parser')
    all_elements_str = [element_to_string(el) for el in soup.find_all()]
    vectorizer = CountVectorizer().fit(all_elements_str)
    target_paths = get_most_similar_paths(current_html, paths, vectorizer)
    return target_paths



def get_most_similar_paths(html, paths, vectorizer):
    """Construct the feature vector according to the hierarchical path and directly find the corresponding element"""
    most_similar_paths = []
    for path in paths:
        most_similar_path = find_most_similar_element_path(html, path, vectorizer)
        if most_similar_path:
            most_similar_paths.append(most_similar_path)
    return most_similar_paths


def generate_element_path(element):
    """Generate the full path of the element"""
    path_parts = []
    while element and element.name:
        path_parts.append(element_to_string(element))
        element = element.parent
    return ' -> '.join(reversed(path_parts))



def find_most_similar_element_path(html, path, vectorizer):
    """Find the element path most similar to the given path in the HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    all_elements = soup.find_all()
    all_elements_paths = [generate_element_path(el) for el in all_elements]
    path_vector = vectorizer.transform([path])
    paths_vectors = vectorizer.transform(all_elements_paths)
    similarities = cosine_similarity(path_vector, paths_vectors)
    most_similar_index = similarities[0].argmax()
    return all_elements_paths[most_similar_index]

def parse_rules_to_paths(rules_json):
    """Extract hierarchical path from JSON rule"""
    paths = []
    for key, value in rules_json.items():
        paths.extend(value)
    return paths


def find_element_by_path(html, path):
    """Find the element in the HTML based on the path"""
    soup = BeautifulSoup(html, 'html.parser')
    current_element = soup

    for part in path.split(" -> "):
        if "[document]" in part:
            continue
        # Split labels and attributes
        match = re.match(r'(\w+)(.*)', part)
        tag = match.group(1) if match else ''
        attr_str = match.group(2).strip() if match else ''
        attrs = parse_attributes(attr_str)
        # Find the next element
        found_element = current_element.find(tag, attrs)
        if found_element:
            current_element = found_element
    return current_element



def parse_attributes(attr_str):
    """Parse the attribute string of the label"""
    attrs = {}
    attributes = split_attributes_improved(attr_str)
    for attribute in attributes:
        if '[' in attribute:  # Check if there is a list in square brackets
            # Use regular expressions to match key-value pairs
            attr_pairs = re.findall(r'(\w+)=\[(.*?)\]|(\w+)=(\S+)', attr_str)
            for attr in attr_pairs:
                if attr[0]:  # Match to the shape like key=[value1, value2] attributes of
                    key, value = attr[0], attr[1].split(',')
                    value = [v.strip(' "[]\'') for v in value]
                else:  # Match to the shape like key=value attributes of
                    key, value = attr[2], attr[3]
                    value = value.strip(' "\'')
                attrs[key] = value
        else:
            # If there is no list, divide it directly by spaces
            attrs_list = re.split(r'\s+', attribute.strip())
            for attr in attrs_list:
                if '=' in attr:
                    key, value = attr.split('=', 1)
                    value = value.strip(' "\'')
                    attrs[key] = value
    return attrs


def split_attributes_improved(attr_str):
    """
    Improved split function for attribute strings.
    This version supports attributes with hyphens and attributes within square brackets.
    It correctly splits attributes such as 'lang=zh data-server-rendered=true data-v-52866abc='.
    """
    # Regular expression to split by space while ignoring spaces inside square brackets and considering hyphens in attribute names
    pattern = r'(\w+(?:-\w+)*=[^\s\[]*(?:\[[^\]]*\])?)'
    attributes = re.findall(pattern, attr_str)
    return attributes

