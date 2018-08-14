#!/usr/bin/env python

import sys
import os
import glob
import json
import zipfile
import fnmatch
import itertools

# okay, first some termini:
# modeldef: json structure like blockstates/*.json
# modelref: json structure of modeldef when a model is referenced
#               ({"model" : "block/block_blah", "x" : 180})
# blockdef: json structure like models/*.json
# variant: a dictionary showing mapping of values to variables

class DirectorySource:
    def __init__(self, path):
        self.path = path

    def glob_files(self, wildcard):
        return [ os.path.relpath(p, self.path) for p in glob.glob(os.path.join(self.path, wildcard)) ]

    def open_file(self, path, mode="r"):
        return open(os.path.join(self.path, path), mode)

    def load_file(self, path):
        f = self.open_file(path)
        data = f.read()
        f.close()
        return data

class JarFileSource:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)

    def glob_files(self, wildcard):
        wildcard = "assets/" + wildcard
        files = []
        for path in self.zip.namelist():
            if fnmatch.fnmatch(path, wildcard):
                files.append(os.path.relpath(path, "assets"))
        return files

    def open_file(self, path, mode="r"):
        return self.zip.open("assets/" + path, mode="r")

    def load_file(self, path):
        f = self.open_file(path)
        data = f.read()
        f.close()
        return data

class Assets:
    def __init__(self, source):
        self.source = None
        if os.path.isdir(source):
            self.source = DirectorySource(source)
        elif os.path.isfile(source) and source.endswith(".jar"):
            self.source = JarFileSource(source)
        else:
            raise RuntimeError("Unknown asset source! Must be asset directory or Minecraft jar file.")

        self.blockstate_base = "minecraft/blockstates"
        self.model_base = "minecraft/models"
        self.texture_base = "minecraft/textures"

        self._model_json_cache = {}
        self._model_cache = {}

    def get_blockstate(self, path):
        filename = os.path.basename(path)
        assert filename.endswith(".json")
        name = filename.replace(".json", "")
        return Blockstate(self, name, json.loads(self.source.load_file(path)))

    @property
    def blockstate_files(self):
        return self.source.glob_files(os.path.join(self.blockstate_base, "*.json"))

    @property
    def blockstates(self):
        blockstates = []
        for path in self.blockstate_files:
            blockstates.append(self.get_blockstate(path))
        return blockstates

    def _get_model_json(self, path):
        if path in self._model_json_cache:
            return self._model_json_cache[path]

        m = json.loads(self.source.load_file(path))

        textures = {}
        elements = {}

        if "parent" in m:
            parent = self._get_model_json(os.path.join(self.model_base, m["parent"] + ".json"))
            textures.update(parent["textures"])
            if "elements" in parent:
                elements = parent["elements"]

        textures.update(m.get("textures", {}))
        if "elements" in m:
            elements = m["elements"]

        modeldef = dict(textures=textures, elements=elements)
        self._model_json_cache[path] = modeldef
        return modeldef

    def get_model(self, path):
        if path in self._model_cache:
            return self._model_cache[path]

        filename = os.path.basename(path)
        assert filename.endswith(".json")
        name = filename.replace(".json", "")
        model = Model(self, name, self._get_model_json(path))
        self._model_cache[path] = model
        return model

    @property
    def model_files(self):
        return self.source.glob_files(os.path.join(self.model_base, "block", "*.json"))

    @property
    def models(self):
        models = []
        for path in self.model_files:
            models.append(self.get_model(path))
        return models

    def load_texture(self, path):
        return self.source.open_file(os.path.join(self.texture_base, path), mode="rb")

class Blockstate:
    def __init__(self, assets, name, data):
        self.assets = assets
        self.name = name
        self.data = data

        self.properties = self._get_properties()
        self.variants = self._get_variants(self.properties)
    
    def evaluate_variant(self, variant):
        modelrefs = []
        if "variants" in self.data:
            for condition, model in self.data["variants"].items():
                condition = parse_variant(condition)
                if is_condition_fulfilled(condition, variant):
                    modelrefs.append(model)
        elif "multipart" in self.data:
            for part in self.data["multipart"]:
                if not "when" in part:
                    modelrefs.append(part["apply"])
                    continue

                when = part["when"]
                if len(when) == 1 and "OR" in when:
                    if any(map(lambda c: is_condition_fulfilled(c, variant), when["OR"])):
                        modelrefs.append(part["apply"])
                else:
                    if is_condition_fulfilled(when, variant):
                        modelrefs.append(part["apply"])
        else:
            assert False, "There must be variants defined!"

        evaluated = []
        for modelref in modelrefs:
            # TODO
            if isinstance(modelref, list):
                modelref = modelref[0]
            model_name = modelref["model"]
            model_transformation = dict(modelref)
            del model_transformation["model"]
            model = self.assets.get_model(os.path.join("minecraft/models", model_name + ".json"))
            evaluated.append((model, model_transformation))
        return evaluated

    def _get_properties(self):
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

        if "variants" in self.data:
            for condition, variant in self.data["variants"].items():
                apply_condition(parse_variant(condition))
        elif "multipart" in self.data:
            for part in self.data["multipart"]:
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

    def _get_variants(self, properties):
        # from a dictionary like {'variable1' : {'value1', 'value2'}}
        # returns all possible variants
        if len(properties) == 0:
            return [{}]

        keys = list(properties.keys())
        values = list(properties.values())

        variants = []
        for product in sorted(itertools.product(*values)):
            variants.append(dict(list(zip(keys, product))))
        return variants

    def __repr__(self):
        return "<Blockstate name=%s>" % self.name

class Model:
    def __init__(self, assets, name, data):
        self.assets = assets
        self.name = name
        self.data = data

    @property
    def textures(self):
        return self.data["textures"]

    @property
    def elements(self):
        return self.data["elements"]

    def load_texture(self, texture):
        if not texture.startswith("#"):
            return self.assets.load_texture(texture + ".png")
            #return os.path.join(TEXTURE_BASE, texture + ".png")
        name = texture[1:]
        if not name in self.textures:
            return None
        return self.load_texture(self.textures[name])

    def __repr__(self):
        return "<Model name=%s>" % self.name

#BASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets/minecraft")
#MODEL_BASE = os.path.join(BASE, "models")
#TEXTURE_BASE = os.path.join(BASE, "textures")

#def resolve_texture(texturesdef, texture):
#    if not texture.startswith("#"):
#        return os.path.join(TEXTURE_BASE, texture + ".png")
#    name = texture[1:]
#    if not name in texturesdef:
#        return None
#    return resolve_texture(texturesdef, texturesdef[name])

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

if __name__ == "__main__":
    assets = Assets(sys.argv[1])

    total_variants = 0
    for blockstate in assets.blockstates:
        total_variants += len(blockstate.variants)
        print(blockstate.name)
        print("Variant properties:", blockstate.properties)
        print("#variants:", len(blockstate.variants))
        for variant in blockstate.variants:
            print(variant)
            for model, transformation in blockstate.evaluate_variant(variant):
                print("=> ", model, transformation)
        print("")
    print("Total variants:", total_variants)
    print("Size: %.2f MB" % (total_variants*32*32*4 / (1024*1024)))

