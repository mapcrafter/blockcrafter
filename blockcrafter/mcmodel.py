# Copyright 2018 Moritz Hilscher
#
# This file is part of Blockcrafter.
#
# Blockcrafter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blockcrafter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Blockcrafter.  If not, see <http://www.gnu.org/licenses/>.

import os
import io
import glob
import json
import zipfile
import fnmatch
import itertools
from PIL import Image

from blockcrafter import util

class BlockstateProperties:
    def __init__(self):
        self.rules = []
    
    def add(self, wildcard, properties):
        self.rules.append((wildcard, properties))

    def get(self, blockstate):
        properties = {}
        for wildcard, p in self.rules:
            if fnmatch.fnmatch(blockstate, wildcard):
                properties.update(p)
        return properties

    @staticmethod
    def load(path):
        properties = BlockstateProperties()
        f = open(path, "r")
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            assert len(parts) == 2, "Invalid line '%s'" % line

            name = parts[0]
            p = parse_variant(parts[1])
            properties.add(name, p)
        f.close()
        return properties

    @staticmethod
    def load_default():
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "blockstates.properties")       
        return BlockstateProperties.load(path)

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

def create_source(path):
    if os.path.isdir(path):
        return DirectorySource(path)
    elif os.path.isfile(path) and (path.endswith(".zip") or path.endswith(".jar")):
        return ZipFileSource(path)
    raise RuntimeError("Unknown asset source! Must be asset directory, Minecraft client jar file, or resource pack zip file.")

def create_builtin_source():
    custom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_assets")
    return create_source(custom_path)

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

class ZipFileSource:
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

def pack_image(image):
    f = io.BytesIO()
    image.save(f, "png")
    return f.getvalue()

# entity textures are often packed together in one texture
# e.g. minecraft/textures/entity/*
# for blockcrafter to be able to use entity textures in custom model files
#          (Minecraft doesn't provide these, we have to build them on our own)
# we have to take these composed texture files apart to single images
# and somehow have them available just like the other block textures
# that's what this class does for a given source
class EntityTextureSource:
    def __init__(self, source):
        self.files = self.create_files(source)
        #for path, data in self.files.items():
        #    print(path)
            #os.makedirs(os.path.dirname("test/" + path), exist_ok=True)
            #f = open("test/" + path, "wb")
            #f.write(data)
            #f.close()

    def create_chest_files(self, source, path):
        base_name = path.replace(".png", "/")
        if len(source.glob_files(path)) == 0:
            return {}
        image = Image.open(source.open_file(path)).convert("RGBA")
        w, h = image.size
        assert w == h
        f = w / 64

        front = image.crop((int(f * 14), int(f * 14), int(f * 28), int(f * 28)))
        front.paste(image.crop((int(f * 14), int(f * 33), int(f * 28), int(f * (33+14)))), (int(f * 0), int(f * 5)))
        side = image.crop((int(f * 0), int(f * 14), int(f * 14), int(f * 28)))
        side.paste(image.crop((int(f * 0), int(f * 33), int(f * 14), int(f * (33+14)))), (int(f * 0), int(f * 5)))

        files = {}
        files[base_name + "front.png"] = pack_image(front)
        files[base_name + "side.png"] = pack_image(side)
        files[base_name + "top.png"] = pack_image(image.crop((int(f * 14), int(f * 0), int(f * 28), int(f * 14))))
        files[base_name + "thing_front.png"] = pack_image(image.crop((int(f * 1), int(f * 1), int(f * 3), int(f * 5))))
        files[base_name + "thing_side.png"] = pack_image(image.crop((int(f * 0), int(f * 1), int(f * 1), int(f * 5))))
        files[base_name + "thing_top.png"] = pack_image(image.crop((int(f * 1), int(f * 0), int(f * 3), int(f * 1))))

        return files

    def create_double_chest_files(self, source, path):
        base_name = path.replace(".png", "/")
        if len(source.glob_files(path)) == 0:
            return {}
        image = Image.open(source.open_file(path)).convert("RGBA")
        w, h = image.size
        assert w == h * 2
        f = w / 128

        left_front = image.crop((int(f * 14), int(f * 14), int(f * 29), int(f * 28)))
        left_front.paste(image.crop((int(f * 14), int(f * 33), int(f * 29), int(f * (33+14)))), (int(f * 0), int(f * 5)))
        right_front = image.crop((int(f * 29), int(f * 14), int(f * 44), int(f * 28)))
        right_front.paste(image.crop((int(f * 29), int(f * 33), int(f * 44), int(f * (33+14)))), (int(f * 0), int(f * 5)))
        side = image.crop((int(f * 0), int(f * 14), int(f * 14), int(f * 28)))
        side.paste(image.crop((int(f * 0), int(f * 33), int(f * 14), int(f * (33+14)))), (int(f * 0), int(f * 5)))
        left_back = image.crop((int(f * 58), int(f * 14), int(f * 73), int(f * 28)))
        left_back.paste(image.crop((int(f * 58), int(f * 33), int(f * 73), int(f * (33+14)))), (int(f * 0), int(f * 5)))
        right_back = image.crop((int(f * 73), int(f * 14), int(f * 88), int(f * 28)))
        right_back.paste(image.crop((int(f * 73), int(f * 33), int(f * 88), int(f * (33+14)))), (int(f * 0), int(f * 5)))

        files = {}
        files[base_name + "left_front.png"] = pack_image(left_front)
        files[base_name + "right_front.png"] = pack_image(right_front)
        files[base_name + "side.png"] = pack_image(side)
        files[base_name + "left_back.png"] = pack_image(left_back)
        files[base_name + "right_back.png"] = pack_image(right_back)
        files[base_name + "left_top.png"] = pack_image(image.crop((int(f * 14), int(f * 0), int(f * 29), int(f * 14))))
        files[base_name + "right_top.png"] = pack_image(image.crop((int(f * 29), int(f * 0), int(f * 44), int(f * 14))))
        files[base_name + "thing_front.png"] = pack_image(image.crop((int(f * 1), int(f * 1), int(f * 3), int(f * 5))))
        files[base_name + "thing_side.png"] = pack_image(image.crop((int(f * 0), int(f * 1), int(f * 1), int(f * 5))))
        files[base_name + "thing_top.png"] = pack_image(image.crop((int(f * 1), int(f * 0), int(f * 3), int(f * 1))))
        return files

    def create_sign_files(self, source, path):
        base_name = path.replace(".png", "/")
        if len(source.glob_files(path)) == 0:
            return {}
        image = Image.open(source.open_file(path)).convert("RGBA")
        w, h = image.size
        assert w == h * 2
        f = w / 64

        files = {}
        files[base_name + "front.png"] = pack_image(image.crop((int(f * 2), int(f * 2), int(f * 24), int(f * 14))))
        files[base_name + "back.png"] = pack_image(image.crop((int(f * 24), int(f * 2), int(f * 46), int(f * 14))))
        files[base_name + "top.png"] = pack_image(image.crop((int(f * 2), int(f * 0), int(f * 24), int(f * 2))))
        files[base_name + "side.png"] = pack_image(image.crop((int(f * 0), int(f * 2), int(f * 2), int(f * 14))))
        files[base_name + "post.png"] = pack_image(image.crop((int(f * 2), int(f * 16), int(f * 4), int(f * 30))))
        return files

    def create_shulker_files(self, source, path):
        base_name = path.replace(".png", "/")
        if len(source.glob_files(path)) == 0:
            return {}
        image = Image.open(source.open_file(path)).convert("RGBA")
        w, h = image.size
        assert w == h
        f = w / 64

        side = image.crop((int(f * 0), int(f * 36), int(f * 16), int(f * 52)))
        part = image.crop((int(f * 0), int(f * 16), int(f * 16), int(f * 28)))
        side.paste(part, (0, 0), part)

        files = {}
        files[base_name + "side.png"] = pack_image(side)
        files[base_name + "top.png"] = pack_image(image.crop((int(f * 16), int(f * 0), int(f * 32), int(f * 16))))
        return files

    def create_files(self, source):
        files = {}
        files.update(self.create_chest_files(source, "minecraft/textures/entity/chest/normal.png"))
        files.update(self.create_chest_files(source, "minecraft/textures/entity/chest/trapped.png"))
        files.update(self.create_chest_files(source, "minecraft/textures/entity/chest/ender.png"))
        files.update(self.create_double_chest_files(source, "minecraft/textures/entity/chest/normal_double.png"))
        files.update(self.create_double_chest_files(source, "minecraft/textures/entity/chest/trapped_double.png"))
        files.update(self.create_sign_files(source, "minecraft/textures/entity/sign.png"))
        for path in source.glob_files("minecraft/textures/entity/shulker/shulker*.png"):
            files.update(self.create_shulker_files(source, path))
        return files

    def glob_files(self, wildcard):
        files = []
        for path in self.files.keys():
            if fnmatch.fnmatch(path, wildcard):
                files.append(path)
        return sorted(files)

    def open_file(self, path, mode="r"):
        return io.BytesIO(self.files[path])

    def load_file(self, path):
        f = self.open_file(path)
        data = f.read()
        f.close()
        return data

class Assets:
    def __init__(self, source):
        self.source = source

        self.blockstate_base = "{prefix}/blockstates"
        self.model_base = "{prefix}/models"
        self.texture_base = "{prefix}/textures"

        self._model_json_cache = {}
        self._model_cache = {}

        self._blockstate_properties = BlockstateProperties.load_default()

    @staticmethod
    def create(asset_paths):
        sources = []
        for path in reversed(asset_paths):
            source = create_source(path)
            sources.append(source)
            sources.append(EntityTextureSource(source))
        sources.insert(0, create_builtin_source())
        return Assets(MultipleSources(sources))

    def get_blockstate(self, identifier):
        prefix, name = identifier.split(":")
        path = os.path.join(self.blockstate_base.format(prefix=prefix), name + ".json")
        properties = self._blockstate_properties.get(identifier)
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

        if "biome_colormap" in self.extra_properties:
            def load_colormap(colormap):
                flipped = False
                if colormap.endswith("_flipped"):
                    flipped = True
                    colormap = colormap.replace("_flipped", "")

                image = Image.open(self.assets.load_texture(self.prefix, "colormap/" + colormap + ".png"))
                image = image.convert("RGBA")
                colors = util.extract_colormap_colors(image)
                return util.encode_colormap_colors(colors)

            colormap = self.extra_properties["biome_colormap"]
            if not "|" in colormap:
                self.extra_properties["biome_colormap"] = load_colormap(colormap)

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

    def resolve_texture(self, texture):
        if not texture.startswith("#"):
            return texture
        name = texture[1:]
        if not name in self.textures:
            return None
        return self.resolve_texture(self.textures[name])

    def load_texture(self, name):
        return self.assets.load_texture(self.prefix, name + ".png")

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

