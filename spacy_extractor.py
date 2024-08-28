import spacy
import re
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.language import Language
import json
from collections import Counter
from itertools import combinations
import time



def load_car_data(file_path):
    with open(file_path, encoding='utf-8-sig') as file:
        data = json.load(file)
    
    car_data = {}
    
    for make, models in data.items():
        make_data = {}
        for model, trims in models.items():
            make_data[model] = list(trims.keys())
        car_data[make] = make_data
    
    return car_data

# file_path = 'test.json'
file_path = r'MakeModelTrimYear.json'
#file_path = r'/home/vuk/CarStoryMakeModelTrim.json'
car_data = load_car_data(file_path)


car_makes = list(car_data.keys())
########################

car_makes.append("mercedes")    # Mercedes-Benz
car_makes.append("chevy")       # Chevrolet
car_makes.append("vw")          # Volkswagen
car_makes.append("rr")          # Rolls Royce
car_makes.append("benz")        # Mercedes-Benz
#######################
all_models = [model for models in car_data.values() for model in models.keys()]
all_trims = [trim for models in car_data.values() for trims in models.values() for trim in trims]

nlp = spacy.load("en_core_web_sm")


@Language.component("add_car_entities")
def add_car_entities(doc):
    matcher = Matcher(nlp.vocab)

    pattern_make = [{"LOWER": {"IN": [make.lower() for make in car_makes]}}]
    matcher.add("CAR_MAKE", [pattern_make])

    matches = matcher(doc)
    spans = [Span(doc, start, end, label=nlp.vocab.strings[match_id]) for match_id, start, end in matches]
    spans = spacy.util.filter_spans(spans)
    doc.ents = spans

    return doc


nlp.add_pipe("add_car_entities", before="ner")

#(O(1) time complexity, possibly lesser accuracy)
# def get_closest_match(text, options):
#     text_set = set(text.lower().split())
#     print(text_set)
#     max_similarity = 0
#     best_match = None

#     for option in options:
#         print(option)
#         option_set = set(option.lower().split())
#         similarity = len(text_set & option_set) / len(text_set | option_set)
        
#         if similarity > max_similarity:
#             max_similarity = similarity
#             best_match = option
#     if max_similarity > 0.01:
#         return best_match
#     return None
    
#     #return best_match   


def sequence_similarity(option, substring):
    option = option.lower()
    substring = substring.lower()
    
    m, n = len(option), len(substring)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if option[i-1] == substring[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n] / max(m, n) 


def get_closest_match(text, options):
    text = re.sub(r'[^a-zA-Z0-9\s#+\-\.!]', ' ', text)
    
    words = text.lower().split()
    max_similarity = 0
    best_matches = []
    all_substrings = []
    
    for i in range(1, min(4, len(words) + 1)):
        for combo in combinations(words, i):
            all_substrings.append(' '.join(combo))
    
    
    for substring in all_substrings:
        for option in options:
            option_lower = option.lower()
            option_chars = Counter(option_lower)
            substring_chars = Counter(substring)

            intersection = sum((option_chars & substring_chars).values())
            union = sum((option_chars | substring_chars).values())

            similarity = intersection / union if union > 0 else 0

            if similarity > max_similarity:
                max_similarity = similarity
                best_matches = [(option, substring)]
            elif similarity == max_similarity:
                best_matches.append((option, substring))

    if max_similarity > 0.40:
        if len(best_matches) > 1:
            best_match = max(best_matches, key=lambda x: (sequence_similarity(x[0], x[1]), len(x[0])))
            return best_match[0]
        else:
            return best_matches[0][0]
    else:
        return None


def extract_info(listing):
    doc = nlp(listing)

    make = model = trim = year = mileage = None

    for ent in doc.ents:
        if ent.label_ == "CAR_MAKE":
            make = ent.text
            break

    if not make:
        make = get_closest_match(listing, car_makes)

    if not make:
        all_models = [model for make_models in car_data.values() for model in make_models]
        model = get_closest_match(listing, all_models)
        if model:
            for m, models in car_data.items():
                if model.lower() in [mod.lower() for mod in models]:
                    make = m
                    break



    if make.lower() == "mercedes" or make == "benz":
        make = "mercedes-benz"
    
    if make.lower() == "vw":
        make = "volkswagen"
    
    if make.lower() == "rr":
        make = "rolls-royce"

    if make.lower() == "chevy":
        make = "chevrolet"

    if make:
        models = list(car_data[make.lower()].keys())
        pattern_model = [{"LOWER": {"IN": [model.lower() for model in models]}}]

        matcher = Matcher(nlp.vocab)
        matcher.add("CAR_MODEL", [pattern_model])

        matches = matcher(doc)
        spans = [Span(doc, start, end, label="CAR_MODEL") for match_id, start, end in matches]
        spans = spacy.util.filter_spans(spans)
        doc.ents = spans
        
        model = get_closest_match(listing, models)
        

        if not model:
            for ent in doc.ents:
                if ent.label_ == "CAR_MODEL":
                    model = ent.text
                    break

        if not model:
            model_to_model_no_space = {model.replace(" ", ""): model for model in models}
            all_models_no_space = list(model_to_model_no_space.keys())
            model_no_space = get_closest_match(listing, all_models_no_space)

            if model_no_space:
                model = model_to_model_no_space[model_no_space]

    if not model and make:
        all_trims = []
        for model_trims in car_data[make.lower()].values():
            all_trims.extend(model_trims)
        
        pattern_trim = [{"LOWER": {"IN": [trim.lower().replace(" ", "") for trim in all_trims]}}]
        
        matcher = Matcher(nlp.vocab)
        matcher.add("CAR_TRIM", [pattern_trim])
        
        matches = matcher(doc)
        spans = [Span(doc, start, end, label="CAR_TRIM") for match_id, start, end in matches]
        spans = spacy.util.filter_spans(spans)
        doc.ents = spans
        trim = get_closest_match(listing, all_trims)
        

        if not trim:
            for ent in doc.ents:
                if ent.label_ == "CAR_TRIM":
                    trim = ent.text
                    break

        if not trim:
            trim_to_trim_no_space = {trim.replace(" ", ""): trim for trim in all_trims}
            all_trims_no_space = list(trim_to_trim_no_space.keys())
            trim_no_space = get_closest_match(listing, all_trims_no_space)

            if trim_no_space:
                trim = trim_to_trim_no_space[trim_no_space]

        if trim and trim in all_trims:
            for model_name, trims in car_data[make.lower()].items():
                if trim.replace(" ", "") in [t.replace(" ", "") for t in trims]:
                    model = model_name
                    break

    elif model:
        trims = car_data[make.lower()][model.lower()]
        pattern_trim = [{"LOWER": {"IN": [trim.lower() for trim in trims]}}]

        matcher = Matcher(nlp.vocab)
        matcher.add("CAR_TRIM", [pattern_trim])

        matches = matcher(doc)
        spans = [Span(doc, start, end, label="CAR_TRIM") for match_id, start, end in matches]
        spans = spacy.util.filter_spans(spans)
        doc.ents = spans

        trim = get_closest_match(listing, trims)

        if not trim:
            for ent in doc.ents:
                if ent.label_ == "CAR_TRIM":
                    trim = ent.text
                    break

        if not trim:
            trim_to_trim_no_space = {trim.replace(" ", ""): trim for trim in trims}
            all_trims_no_space = list(trim_to_trim_no_space.keys())
            trim_no_space = get_closest_match(listing, all_trims_no_space)

            if trim_no_space:
                trim = trim_to_trim_no_space[trim_no_space]

    if not model:
        all_models = []
        for make_data in car_data.values():
            all_models.extend(make_data.keys())

        pattern_model = [{"LOWER": {"IN": [model.lower().replace(" ", "") for model in all_models]}}]
        matcher = Matcher(nlp.vocab)
        matcher.add("CAR_MODEL", [pattern_model])
        matches = matcher(doc)
        spans = [Span(doc, start, end, label="CAR_MODEL") for match_id, start, end in matches]
        spans = spacy.util.filter_spans(spans)
        doc.ents = spans

        found_model = get_closest_match(listing, all_models)

        if not found_model:
            for ent in doc.ents:
                if ent.label_ == "CAR_MODEL":
                    found_model = ent.text
                    break

        if not found_model:
            model_to_model_no_space = {model.replace(" ", ""): model for model in all_models}
            all_models_no_space = list(model_to_model_no_space.keys())
            model_no_space = get_closest_match(listing, all_models_no_space)

            if model_no_space:
                found_model = model_to_model_no_space[model_no_space]

        if found_model:
            model = found_model  # Only set model if it wasn't already set
            if not make:
                for m, models in car_data.items():
                    if model.lower() in [m.lower() for m in models]:
                        make = m
                        break

        if not model:
            all_trims = []
            for make_data in car_data.values():
                for model_data in make_data.values():
                    all_trims.extend(model_data)
            
            pattern_trim = [{"LOWER": {"IN": [trim.lower().replace(" ", "") for trim in all_trims]}}]
            matcher = Matcher(nlp.vocab)
            matcher.add("CAR_TRIM", [pattern_trim])
            matches = matcher(doc)
            spans = [Span(doc, start, end, label="CAR_TRIM") for match_id, start, end in matches]
            spans = spacy.util.filter_spans(spans)
            doc.ents = spans

            found_trim = get_closest_match(listing, all_trims)

            if not found_trim:
                for ent in doc.ents:
                    if ent.label_ == "CAR_TRIM":
                        found_trim = ent.text
                        break

            if not found_trim:
                trim_to_trim_no_space = {trim.replace(" ", ""): trim for trim in all_trims}
                all_trims_no_space = list(trim_to_trim_no_space.keys())
                trim_no_space = get_closest_match(listing, all_trims_no_space)

                if trim_no_space:
                    found_trim = trim_to_trim_no_space[trim_no_space]

            if found_trim:
                trim = found_trim  # Only set trim if it wasn't already set
                # Don't overwrite existing make or model based on trim

    if trim and not model:
        for make_name, models in car_data.items():
            for model_name, trims in models.items():
                if trim.lower() in [t.lower() for t in trims]:
                    if not model:
                        model = model_name
                    if not make:
                        make = make_name
                    break

    if model and not make:
        for make_name, models in car_data.items():
            if model.lower() in [m.lower() for m in models.keys()]:
                make = make_name
                break
            
    
    if make and model:
        if model.lower() not in car_data[make.lower()]:
            model = None
            trim = None
        elif trim:
            if trim.lower() not in [t.lower() for t in car_data[make.lower()][model.lower()]]:
                trim = None
        else:
            trims = car_data[make.lower()][model.lower()]
            found_trim = get_closest_match(listing, trims)
            if found_trim:
                trim = found_trim

    year_match = re.search(r'\b(?:19|20)\d{2}\b(?!\s*(?:miles|mi))', listing)
    if year_match:
        year = int(year_match.group(0))

    mileage_pattern = r'\b(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s*(?:miles|mi\.?|k miles|k mi\.?|k|km)\b'
    mileage_match = re.search(mileage_pattern, listing, re.IGNORECASE)
    if mileage_match:
        mileage_str = mileage_match.group(1).replace(',', '')
        mileage = int(float(mileage_str) * (1000 if 'k' in mileage_match.group(0).lower() else 1))


    
    return_dict =  {
        'Make': make.lower(),
        'Model': model,
        'Trim': trim,
        'Year': year,
        'Mileage': mileage
    }

    for key, value in return_dict.items():
        if value == None:
            return_dict[key] = " "

    return return_dict 


