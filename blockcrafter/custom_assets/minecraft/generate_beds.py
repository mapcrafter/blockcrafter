#!/usr/bin/env python3

colors = [
    "white",
    "orange",
    "magenta",
    "light_blue",
    "yellow",
    "lime",
    "pink",
    "gray",
    "light_gray",
    "cyan",
    "purple",
    "blue",
    "brown",
    "green",
    "red",
    "black"
]

for color in colors:
    name = "%s_bed" % color

    f = open("models/block/%s_head.json" % name, "w")
    f.write("""{{
    "parent": "block/base_bed_head",
    "textures": {{
        "head_top": "entity/bed/{color}/head_top",
        "head_front": "entity/bed/{color}/head_front",
        "head_side": "entity/bed/{color}/head_side",
        "foot_top": "entity/bed/{color}/foot_top",
        "foot_front": "entity/bed/{color}/foot_front",
        "foot_side": "entity/bed/{color}/foot_side",
        "stand_outer": "entity/bed/{color}/stand_outer",
        "stand_inner": "entity/bed/{color}/stand_inner"
    }}
    }}""".format(color=color))
    
    f = open("models/block/%s_foot.json" % name, "w")
    f.write("""{{
    "parent": "block/base_bed_foot",
    "textures": {{
        "head_top": "entity/bed/{color}/head_top",
        "head_front": "entity/bed/{color}/head_front",
        "head_side": "entity/bed/{color}/head_side",
        "foot_top": "entity/bed/{color}/foot_top",
        "foot_front": "entity/bed/{color}/foot_front",
        "foot_side": "entity/bed/{color}/foot_side",
        "stand_outer": "entity/bed/{color}/stand_outer",
        "stand_inner": "entity/bed/{color}/stand_inner"
    }}
    }}""".format(color=color))
    
    f = open("blockstates/%s.json" % name, "w")
    f.write("""{{
    "variants": {{
        "part=head,facing=north": {{ "model": "block/{name}_head", "y": 0 }},
        "part=head,facing=south": {{ "model": "block/{name}_head", "y": 180 }},
        "part=head,facing=east": {{ "model": "block/{name}_head", "y": 90 }},
        "part=head,facing=west": {{ "model": "block/{name}_head", "y": 270 }},

        "part=foot,facing=north": {{ "model": "block/{name}_foot", "y": 0 }},
        "part=foot,facing=south": {{ "model": "block/{name}_foot", "y": 180 }},
        "part=foot,facing=east": {{ "model": "block/{name}_foot", "y": 90 }},
        "part=foot,facing=west": {{ "model": "block/{name}_foot", "y": 270 }}
    }}
    }}""".format(name=name))
    f.close()

