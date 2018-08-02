#!/usr/bin/env python

import sys
import os
import json
import itertools

# okay, first some termini:
# modeldef: json structure like blockstates/*.json
# modelref: json structure of modeldef when a model is referenced
#               ({"model" : "block/block_blah", "x" : 180})
# blockdef: json structure like models/*.json
# variant: a dictionary showing mapping of values to variables

BASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets/minecraft")
MODEL_BASE = os.path.join(BASE, "models")
TEXTURE_BASE = os.path.join(BASE, "textures")

model_cache = {}
def load_modeldef(path):
    global model_cache
    if path in model_cache:
        return model_cache[path]

    m = json.load(open(path))

    textures = {}
    elements = {}

    if "parent" in m:
        parent = load_modeldef(os.path.join(MODEL_BASE, m["parent"] + ".json"))
        textures.update(parent["textures"])
        if "elements" in parent:
            elements = parent["elements"]

    textures.update(m.get("textures", {}))
    if "elements" in m:
        elements = m["elements"]

    modeldef = dict(textures=textures, elements=elements)
    model_cache[path] = modeldef
    return modeldef

def resolve_texture(texturesdef, texture):
    if not texture.startswith("#"):
        return os.path.join(TEXTURE_BASE, texture + ".png")
    name = texture[1:]
    if not name in texturesdef:
        return None
    return resolve_texture(texturesdef, texturesdef[name])

def resolve_element(texturesdef, elementdef):
    faces = {}
    for direction, facedef in elementdef["faces"].items():
        faces[side] = resolve_texture(textures, face["texture"])
    return faces

def is_complete(modeldef):
    textures = set()
    for elementdef in modeldef["elements"]:
        for direction, facedef in elementdef["faces"].items():
            textures.add(facedef["texture"])
    return all([ resolve_texture(textures, t) is not None for t in textures ])

def load_blockdef(path):
    m = json.load(open(path))
    return m

def parse_variant(condition):
    if condition == "":
        return {}
    return dict(map(lambda pair: pair.split("="), condition.split(",")))

def encode_variant(variant):
    items = list(variant.items())
    items.sort(key = lambda i: i[0])
    return ",".join(map(lambda i: "=".join(i), items))

def is_condition_fulfilled(condition, variant):
    # condition and variant are both dictionaries
    # key => value mean that variable 'key' has value 'value'
    # ==> variant variables must have same values as condition
    # some litte quirks:
    # - sometimes values are of type bool (from json) => make that to a string
    # - values in condition may be of form 'value1|value2' => means that
    #     values 'value1' and 'value2' are acceptable
    for key, value in condition.items():
        if not key in variant:
            return False

        if type(value) == bool:
            value = "true" if value else "false"

        values = set([value])
        if "|" in value:
            values = set(value.split("|"))
        if not variant[key] in values:
            return False
    return True

def get_blockdef_variables(blockdef):
    variables = {}

    def apply_condition(condition):
        nonlocal variables
        for key, value in condition.items():
            if key not in variables:
                variables[key] = set()
            
            if type(value) == bool:
                value = "true" if value else "false"

            values = set([value])
            if "|" in value:
                values = set(value.split("|"))
            # TODO this is a bit hacky
            # (just assume there must be true to false value, and vice versa)
            if "true" in values:
                values.add("false")
            if "false" in values:
                values.add("true")
            variables[key].update(values)

    if "variants" in blockdef:
        for condition, variant in blockdef["variants"].items():
            apply_condition(parse_variant(condition))
    elif "multipart" in blockdef:
        for part in blockdef["multipart"]:
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

def get_variable_variants(variables):
    # from a dictionary like {'variable1' : {'value1', 'value2'}}
    # returns all possible variants
    if len(variables) == 0:
        return [{}]
    
    keys = list(variables.keys())
    values = list(variables.values())

    variants = []
    for product in itertools.product(*values):
        variants.append(dict(list(zip(keys, product))))
    return variants

def get_blockdef_variants(blockdef):
    return get_variable_variants(get_blockdef_variables(blockdef))

def get_blockdef_modelrefs(blockdef, variant):
    models = []
    if "variants" in blockdef:
        for condition, model in blockdef["variants"].items():
            condition = parse_variant(condition)
            if is_condition_fulfilled(condition, variant):
                models.append(model)
    elif "multipart" in blockdef:
        for part in blockdef["multipart"]:
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
    #    m = load_modeldef(path)
    #    print(path, is_complete(m))
    #    print(m["elements"])
    #    print(resolve_element(m["elements"][0], m["textures"]))

    total_variants = 0
    for path in sys.argv[1:]:
        blockdef = load_blockdef(path)
        variables = get_blockdef_variables(blockdef)
        variants = get_blockdef_variants(blockdef)
        total_variants += len(variants)
        print(path)
        print("Variant variables:", variables)
        print("#variants:", len(variants))
        for variant in variants:
            print(variant)
            for modelrefs in get_blockdef_modelrefs(blockdef, variant):
                print("=> ", modelrefs)
        print("")
    print("Total variants:", total_variants)
    print("Size: %.2f MB" % (total_variants*32*32*4 / (1024*1024)))

