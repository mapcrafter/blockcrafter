#!/usr/bin/env python

import sys
import os
import json
import itertools

BASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets/minecraft")
MODEL_BASE = os.path.join(BASE, "models")
TEXTURE_BASE = os.path.join(BASE, "textures")

model_cache = {}
def load_model(path):
    global model_cache
    if path in model_cache:
        return model_cache[path]

    m = json.load(open(path))

    textures = {}
    elements = {}

    if "parent" in m:
        parent = load_model(os.path.join(MODEL_BASE, m["parent"] + ".json"))
        textures.update(parent["textures"])
        if "elements" in parent:
            elements = parent["elements"]

    textures.update(m.get("textures", {}))
    if "elements" in m:
        elements = m["elements"]

    model = dict(textures=textures, elements=elements)
    #print("Loaded:", path, model)
    model_cache[path] = model
    return model

def resolve_texture(texture, textures):
    if not texture.startswith("#"):
        return os.path.join(TEXTURE_BASE, texture + ".png")
    name = texture[1:]
    if not name in textures:
        return None
    return resolve_texture(textures[name], textures)

def resolve_element(element, textures):
    faces = {}
    for side, face in element["faces"].items():
        faces[side] = resolve_texture(face["texture"], textures)
    return faces

def is_complete(model):
    textures = set()
    for element in model["elements"]:
        for side, face in element["faces"].items():
            textures.add(face["texture"])
    #print("textures", textures)
    return all([ resolve_texture(t, textures) is not None for t in textures ])

def load_blockstate(path):
    m = json.load(open(path))
    return m

def parse_condition(condition):
    if condition == "":
        return {}
    return dict(map(lambda pair: pair.split("="), condition.split(",")))
def is_condition_fulfilled(condition, variant):
    for key, value in condition.items():
        if not key in variant:
            return False
        
        # HMM hack
        if type(value) == bool:
            value = "true" if value else "false"

        values = set([value])
        if "|" in value:
            values = set(value.split("|"))
        if not variant[key] in values:
            return False
    return True

def get_blockstate_domain(blockstate):
    variables = {}

    def apply_condition(condition):
        nonlocal variables
        for key, value in condition.items():
            if key not in variables:
                variables[key] = set()
            
            # HMM hack
            if type(value) == bool:
                value = "true" if value else "false"

            values = set([value])
            if "|" in value:
                values = set(value.split("|"))
            # HMM hack
            if "true" in values:
                values.add("false")
            if "false" in values:
                values.add("true")
            variables[key].update(values)

    if "variants" in blockstate:
        for condition, variant in blockstate["variants"].items():
            apply_condition(parse_condition(condition))
    elif "multipart" in blockstate:
        for part in blockstate["multipart"]:
            if not "when" in part:
                continue
            when = part["when"]
            if len(when) == 1 and "OR" in when:
                conditions = when["OR"]
                for condition in conditions:
                    apply_condition(condition)
            else:
                apply_condition(when)
    else:
        assert False, "There must be variants defined!"
    return variables

def get_domain_variants(domain):
    # from a dictionary like {'variable1' : {'value1', 'value2'}}
    # returns all possible variants
    if len(domain) == 0:
        return [{}]
    
    keys = list(domain.keys())
    values = list(domain.values())

    variants = []
    for product in itertools.product(*values):
        variants.append(dict(list(zip(keys, product))))
    return variants

def get_blockstate_models(blockstate, variant):
    models = []
    if "variants" in blockstate:
        for condition, model in blockstate["variants"].items():
            condition = parse_condition(condition)
            if is_condition_fulfilled(condition, variant):
                models.append(model)
    elif "multipart" in blockstate:
        for part in blockstate["multipart"]:
            if not "when" in part:
                models.append(part["apply"])
                continue

            when = part["when"]
            if len(when) == 1 and "OR" in when:
                if any(map(lambda c: is_condition_fulfilled(c, variant), when["OR"])):
                    models.append(part["apply"])
            else:
                if is_condition_fulfilled(when, variant):
                    models.append(part["apply"])
    else:
        assert False, "There must be variants defined!"
    return models

if __name__ == "__main__":
    #for path in sys.argv[1:]:
    #    m = load_model(path)
    #    print(path, is_complete(m))
    #    print(m["elements"])
    #    print(resolve_element(m["elements"][0], m["textures"]))

    import functools
    total_variants = 0
    for path in sys.argv[1:]:
        blockstate = load_blockstate(path)
        domain = get_blockstate_domain(blockstate)
        variants = functools.reduce(lambda x, y: x * y, map(lambda v: len(v), domain.values()), 1)
        total_variants += variants
        print(path)
        print("Domain:", domain)
        print("#variants:", variants)
        for variant in get_domain_variants(domain):
            print(variant)
            for model in get_blockstate_models(blockstate, variant):
                print("\t", model)
        print("")
    print("Total variants:", total_variants)
    print("Size: %.2f MB" % (total_variants*32*32*4 / (1024*1024)))
