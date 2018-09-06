#!/usr/bin/env python

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

def load_blockstate_properties():
    properties = {}

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "blockstates.properties")
    f = open(path, "r")
    for line in f.readlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ")
        assert len(parts) == 2, "Invalid line '%s'" % line

        name = parts[0]
        p = parse_variant(parts[1])
        properties[name] = p
    f.close()
    return properties

class DirectorySource:
    def __init__(self, path):
        self.path = path

    def glob_files(self, wildcard):
        return sorted([ os.path.relpath(p, self.path) for p in glob.glob(os.path.join(self.path, wildcard)) ])

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
        return sorted(files)

    def open_file(self, path, mode="r"):
        return self.zip.open("assets/" + path, mode="r")

    def load_file(self, path):
        f = self.open_file(path)
        data = f.read()
        f.close()
        return data

class MultipleSources:
    def __init__(self, sources):
        self.sources = sources

    def glob_files(self, wildcard):
        files = []
        for source in self.sources:
            files.extend(source.glob_files(wildcard))
        return sorted(list(set(files)))

    def open_file(self, path, mode="r"):
        for source in self.sources:
            try:
                f = source.open_file(path, mode)
                return f
            except Exception as e:
                pass
        raise RuntimeError("Unable to find file '%s' in any source!" % path)

    def load_file(self, path):
        f = self.open_file(path)
        data = f.read()
        f.close()
        return data

class Assets:
    def __init__(self, path):
        source = None
        if os.path.isdir(path):
            source = DirectorySource(path)
        elif os.path.isfile(path) and path.endswith(".jar"):
            source = JarFileSource(path)
        else:
            raise RuntimeError("Unknown asset source! Must be asset directory or Minecraft jar file.")

        # always also include some custom assets
        custom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_assets")
        self.source = MultipleSources([DirectorySource(custom_path), source])

        self.blockstate_base = "{prefix}/blockstates"
        self.model_base = "{prefix}/models"
        self.texture_base = "{prefix}/textures"

        self._model_json_cache = {}
        self._model_cache = {}

        self._blockstate_properties = load_blockstate_properties()

    def get_blockstate(self, identifier):
        prefix, name = identifier.split(":")
        path = os.path.join(self.blockstate_base.format(prefix=prefix), name + ".json")
        properties = self._blockstate_properties.get(identifier, {})
        return Blockstate(self, prefix, name, json.loads(self.source.load_file(path)), properties=properties)

    @property
    def blockstate_files(self):
        return self.source.glob_files(os.path.join(self.blockstate_base.format(prefix="*"), "*.json"))

    @property
    def blockstates(self):
        blockstates = []
        for path in self.blockstate_files:
            filename = os.path.basename(path)
            assert filename.endswith(".json")
            name = filename.replace(".json", "")
            prefix = path.split("/")[0]
            blockstates.append(self.get_blockstate(prefix + ":" + name))
        return blockstates

    def _get_model_json(self, path):
        if path in self._model_json_cache:
            return self._model_json_cache[path]

        m = json.loads(self.source.load_file(path))

        textures = {}
        elements = {}

        if "parent" in m:
            prefix = path.split("/")[0]
            parent = self._get_model_json(os.path.join(self.model_base.format(prefix=prefix), m["parent"] + ".json"))
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
        prefix = path.split("/")[0]
        model = Model(self, prefix, name, self._get_model_json(path))
        self._model_cache[path] = model
        return model

    @property
    def model_files(self):
        return self.source.glob_files(os.path.join(self.model_base.format(prefix="*"), "block", "*.json"))

    @property
    def models(self):
        models = []
        for path in self.model_files:
            models.append(self.get_model(path))
        return models

    def load_texture(self, prefix, path):
        return self.source.open_file(os.path.join(self.texture_base.format(prefix=prefix), path), mode="rb")

class Blockstate:
    def __init__(self, assets, prefix, name, data, properties={}):
        self.assets = assets
        self.prefix = prefix
        self.name = name
        self.data = data

        self.extra_properties = properties
        self.waterloggable = properties.get("is_waterloggable", "") == "true"
        self.inherently_waterlogged = properties.get("inherently_waterlogged", "") == "true"

        self.properties = self._get_properties()
        if self.waterloggable:
            self.properties["waterlogged"] = ["true", "false"]
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
            model = self.assets.get_model(os.path.join(self.prefix + "/models", model_name + ".json"))
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
            variant = dict(list(zip(keys, product)))
            if self.waterloggable and self.inherently_waterlogged:
                if variant["waterlogged"] == "true":
                    del variant["waterlogged"]
            variants.append(variant)
        return variants

    def __repr__(self):
        return "<Blockstate prefix=%s name=%s>" % (self.prefix, self.name)

class Model:
    def __init__(self, assets, prefix, name, data):
        self.assets = assets
        self.prefix = prefix
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
            return self.assets.load_texture(self.prefix, texture + ".png")
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
    if len(variant) == 0:
        return "-"
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

