#!/usr/bin/env python3

for color in ["", "white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"]:
    block_name = "%s_shulker_box" % color
    texture_name = "shulker_%s" % color
    if color == "":
        block_name = "shulker_box"
        texture_name = "shulker"

    f = open("models/block/%s.json" % block_name, "w")
    f.write("""{{
    "parent" : "block/base_shulker_box",
    "textures":  {{
        "side": "entity/shulker/{texture_name}/side",
        "top": "entity/shulker/{texture_name}/top"
    }}
    }}""".format(texture_name=texture_name))
    f.close()

    f = open("blockstates/%s.json" % block_name, "w")
    f.write("""{{
    "variants": {{
        "facing=north" : {{ "model" : "block/{block_name}", "x": 90 }},
        "facing=south" : {{ "model" : "block/{block_name}", "x": 270 }},
        "facing=east" : {{ "model" : "block/{block_name}", "z": 90 }},
        "facing=west" : {{ "model" : "block/{block_name}", "z": 270 }},
        "facing=up" : {{ "model" : "block/{block_name}" }},
        "facing=down" : {{ "model" : "block/{block_name}", "x" : 180 }}
    }}
    }}""".format(block_name=block_name))
    f.close()
