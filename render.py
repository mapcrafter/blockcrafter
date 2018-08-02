import os
import numpy as np
import math
import time
from PIL import Image
from glumpy import app, gl, glm, gloo, data
from glumpy import transforms

import mcmodel

def geom_cube():
    vtype = [('a_position', np.float32, 3),
             ('a_texcoord', np.float32, 2),
             ('a_normal',   np.float32, 3)]
    itype = np.uint32

    # Vertices positions
    p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                  [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=float)
    # Face Normals
    n = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0],
                  [-1, 0, 1], [0, -1, 0], [0, 0, -1]])
    # Texture coords
    t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    faces_p = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_n = [0, 0, 0, 0,  1, 1, 1, 1,   2, 2, 2, 2,
               3, 3, 3, 3,  4, 4, 4, 4,   5, 5, 5, 5]
    faces_t = [0, 1, 2, 3,  0, 1, 2, 3,   0, 1, 2, 3,
               3, 2, 1, 0,  0, 1, 2, 3,   0, 1, 2, 3]

    vertices = np.zeros(24, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal']   = n[faces_n]
    #vertices['a_texcoord'] = t[faces_t]

    filled = np.resize(
       np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
    filled += np.repeat(4 * np.arange(6, dtype=itype), 6)

    outline = np.resize(
        np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=itype), 6 * (2 * 4))
    outline += np.repeat(4 * np.arange(6, dtype=itype), 8)
    vertices = vertices.view(gloo.VertexBuffer)
    filled   = filled.view(gloo.IndexBuffer)

    return vertices, filled

vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
attribute vec3 a_normal;        // Vertex normal
//attribute vec2 a_texcoord;      // Vertex texture coordinates
varying vec3   v_normal;        // Interpolated normal (out)
varying vec3   v_position;      // Interpolated position (out)
varying vec3   v_texcoord;      // Interpolated fragment texture coordinates (out)

void main()
{
    // Assign varying variables
    v_normal   = a_normal;
    v_position = a_position;
    //v_texcoord = a_texcoord;
    v_texcoord = a_position;

    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""

fragment = """
uniform mat4      u_model;           // Model matrix
uniform mat4      u_view;            // View matrix
uniform mat4      u_normal;          // Normal matrix
uniform mat4      u_projection;      // Projection matrix
uniform samplerCube u_texture;         // Texture 
uniform vec3      u_light_position;  // Light position
uniform vec3      u_light_intensity; // Light intensity

varying vec3      v_normal;          // Interpolated normal (in)
varying vec3      v_position;        // Interpolated position (in)
varying vec3      v_texcoord;        // Interpolated fragment texture coordinates (in)
void main()
{
    // Calculate normal in world coordinates
    vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

    // Calculate the location of this fragment (pixel) in world coordinates
    vec3 position = vec3(u_view*u_model * vec4(v_position, 1));

    // Calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = u_light_position - position;

    // Calculate the cosine of the angle of incidence (brightness)
    float brightness = dot(normal, surfaceToLight) /
                      (length(surfaceToLight) * length(normal));
    brightness = max(min(brightness,1.0),0.0);

    // Calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)

    // Get texture color
    vec4 t_color = textureCube(u_texture, v_texcoord);

    // Final color
    //gl_FragColor = vec4(t_color.rgb * (0.1 + 0.9*brightness * u_light_intensity), t_color.a);
    gl_FragColor = t_color;
}
"""

VERTEX = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
attribute vec3 a_normal;        // Vertex normal
attribute vec2 a_texcoord;      // Vertex texture coordinates
varying vec3   v_normal;        // Interpolated normal (out)
varying vec3   v_position;      // Interpolated position (out)
varying vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)
varying vec3 v_fragpos;

void main()
{
    // Assign varying variables
    v_normal = (u_model * vec4(a_normal, 1.0)).xyz;
    v_fragpos = (u_model, vec4(a_position, 1.0)).xyz;

    v_position = a_position;
    v_texcoord = a_texcoord;

    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
"""

FRAGMENT = """
uniform mat4      u_model;           // Model matrix
uniform mat4      u_view;            // View matrix
uniform mat4      u_normal;          // Normal matrix
uniform mat4      u_projection;      // Projection matrix
uniform sampler2D u_texture;         // Texture 
uniform vec3      u_light_position;  // Light position
uniform vec3      u_light_intensity; // Light intensity

varying vec3      v_normal;          // Interpolated normal (in)
varying vec3      v_position;        // Interpolated position (in)
varying vec2      v_texcoord;        // Interpolated fragment texture coordinates (in)
varying vec3 v_fragpos;
void main()
{
    // Calculate normal in world coordinates
    //mat4 u_normal = inverse(u_view * u_model);
    vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

    // Calculate the location of this fragment (pixel) in world coordinates
    vec3 position = vec3(u_view*u_model * vec4(v_position, 1));

    // Calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = u_light_position - position;

    // Calculate the cosine of the angle of incidence (brightness)
    float brightness = dot(normal, surfaceToLight) /
                      (length(surfaceToLight) * length(normal));
    brightness = max(min(brightness,1.0),0.0);

    // Calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)

    // Get texture color
    vec4 t_color = texture2D(u_texture, v_texcoord);

    // Final color
    if (t_color.a <= 0.001) {
        discard;
    }
    //gl_FragColor = vec4(t_color.rgb * (0.5 + 0.4*brightness * u_light_intensity), t_color.a);
    //gl_FragColor = t_color;

    gl_FragColor = vec4((v_normal + 1.0) / 2.0, 1.0);


    vec3 norm = normalize(v_normal);
    vec3 lightDir = normalize(u_light_position - v_fragpos);

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 diffuse = diff * lightColor;

    gl_FragColor = vec4(t_color.rgb * (0.6 + 0.4 * diffuse), t_color.a);
}
"""

class CubemapCube(gloo.Program):
    def __init__(self, cubemap, vertex=vertex, fragment=fragment, light=(-2, 2, 2)):
        super().__init__(vertex, fragment)

        self._vertices, self._indices = geom_cube()
        self.bind(self._vertices)

        self["u_texture"] = cubemap

        self["u_model"] = self.model
        self["u_view"] = glm.translation(0, 0, -5)
        self["u_projection"] = np.eye(4, dtype=np.float32)

        self["u_light_position"] = light
        self["u_light_intensity"] = 1.0, 1.0, 1.0

    @property
    def model(self):
        return np.eye(4, dtype=np.float32)

    def render(self, model):
        self.draw(gl.GL_TRIANGLES, self._indices)

def cube_side(i):
    sides = [
        (90, 0, 1, 0), # pos x
        (-90, 0, 1, 0), # neg x
        (-90, 1, 0, 0), # pos y
        (90, 1, 0, 0), # neg y
        (0, 0, 0, 1), # pos z
        (180, 0, 1, 0), # neg z
    ]

    normals = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    transform = np.eye(4, dtype=np.float32)
    glm.translate(transform, 0, 0, 1)
    glm.rotate(transform, *sides[i])
    return np.array(normals[i], dtype=np.float32), transform

class TextureQuad(gloo.Program):
    def __init__(self, vertex=VERTEX, fragment=FRAGMENT, texcoords=None, model=np.eye(4, dtype=np.float32), texture=None):
        super().__init__(vertex, fragment, count=4)

        if texcoords is None:
            texcoords = [(0, 1), (0, 0), (1, 1), (1, 0)]

        self["a_position"] = [(-1,-1, 0), (-1,+1, 0), (+1,-1, 0), (+1,+1, 0)]
        self["a_texcoord"] = texcoords

        # TODO
        self["a_normal"] = np.array([0.0, 0.0, 1.0])
        self["u_light_position"] = -1.0, 2.0, 5.0

        self.texture = texture
        #self.texture.interpolation = gl.GL_LINEAR
        self.model = model

    def render(self, model, view, projection):
        self["u_texture"] = self.texture
        self["u_model"] = np.dot(self.model, model)
        self["u_view"] = view
        self["u_projection"] = projection
        self["u_light_position"] = -1.0, 2.0, 5.0
        #self["u_normal"] = np.array(np.matrix(np.dot(view, np.dot(self.model, model))).I.T)
        self.draw(gl.GL_TRIANGLE_STRIP)

class Cube:
    def __init__(self, sides):
        self.faces = []
        for i, side in enumerate(sides):
            normal, transform = cube_side(i)
            texture = side[0].view(gloo.Texture2D)
            #if i == 4:
            #    # top texture must actually be a bit rotated
            #    glm.rotate(transform, -90, 0, 0, 1)
            #    #texture.interpolation = gl.GL_LINEAR
            #    pass
            #if i == 3:
            #    # bottom texture must actually be a bit rotated
            #    glm.rotate(transform, 90, 0, 1, 0)
            #if i != 2 and i != 3:
            #    continue
            uv0, uv1 = side[1]
            texcoords = np.array([(uv0[0], uv1[1]), uv0, uv1, (uv1[0], uv0[1])], dtype=np.float32)
            glm.translate(transform, 0.0, 0.0, 0.0)
            self.faces.append(TextureQuad(model=transform, texture=texture, texcoords=texcoords))

    def render(self, model, view, projection):
        for face in self.faces:
            face.render(model, view, projection)

class Element(Cube):
    def __init__(self, elementdef, texturesdef):
        super().__init__(Element.load_faces(elementdef, texturesdef))

        self.xyz0 = (np.array(elementdef["from"]) / 16.0 * 2.0) - 1.0
        self.xyz1 = (np.array(elementdef["to"]) / 16.0 * 2.0) - 1.0
        self.rotation = elementdef.get("rotation", None)

    @property
    def model(self):
        model = np.eye(4, dtype=np.float32)

        # apply from/to attributes
        scale = (self.xyz1 - self.xyz0) * 0.5
        translate = (self.xyz0 + self.xyz1) / 2.0
        glm.scale(model, scale[0], scale[1], scale[2])
        glm.translate(model, translate[0], translate[1], translate[2])
 
        # apply rotation attributes
        if self.rotation:
            axis = {"x" : [1, 0, 0],
                    "y" : [0, 1, 0],
                    "z" : [0, 0, 1]}[self.rotation["axis"]]
            origin = (np.array(self.rotation.get("origin", [8, 8, 8]), dtype=np.float32) - 8.0) / 16.0 * 2.0
            origin *= -1.0
            glm.translate(model, *origin)
            glm.rotate(model, self.rotation["angle"], *axis)
            origin *= -1.0
            glm.translate(model, *origin)

        return model

    def render(self, model, view, projection):
        super().render(np.dot(self.model, model), view, projection)

    @staticmethod
    def load_faces(elementdef, texturesdef):
        # order of minecraft directions to order of cube sides
        mc_to_opengl = [
            "east",   # pos x
            "west",   # neg x
            "up",     # pos y
            "down",   # neg y
            "south",  # pos z
            "north",  # neg z
        ]

        faces = {}
        for direction, facedef in elementdef["faces"].items():
            path = mcmodel.resolve_texture(texturesdef, facedef["texture"])
            if path is None:
                raise RuntimeError("Face in direction '%s' has no texture associated" % direction)
            uvs = np.array(facedef.get("uv", [0, 0, 16, 16]), dtype=np.float32) / 16.0
            uv0, uv1 = uvs[:2], uvs[2:]
            faces[direction] = (np.array(Image.open(path)), (uv0, uv1))

        # gather faces in order for cube sides
        sides = [ faces.get(direction, None) for direction in mc_to_opengl ]
        # get first side that is not empty
        empty = (np.zeros((1, 1, 4), dtype=np.uint8), ((0, 0), (16, 16)))
        sides = [ (s if s is not None else empty) for s in sides ]
        assert len(sides) == 6
        return sides

class Model:
    def __init__(self, modeldef):
        self.elements = []
        for elementdef in modeldef["elements"]:
            self.elements.append(Element(elementdef, modeldef["textures"]))

    def render(self, model, view, projection, modelref={}, rotation=0):
        m = np.eye(4, dtype=np.float32)
        if "x" in modelref:
            glm.rotate(m, modelref["x"], 1, 0, 0)
        if "y" in modelref:
            glm.rotate(m, modelref["y"], 0, 1, 0)
        model = np.dot(m, model)

        if "uvlock" in modelref:
            # TODO
            pass

        for element in self.elements:
            element.render(model, view, projection)

class Block:
    def __init__(self, blockdef):
        self.blockdef = blockdef
        self.models = {}
        self.variants = {}

    def _load_modeldef(self, name):
        modeldef = mcmodel.load_modeldef(os.path.join(mcmodel.MODEL_BASE, name + ".json"))
        model = Model(modeldef)
        return model

    def _load_variant(self, variant):
        modelrefs = []
        for modelref in mcmodel.get_blockdef_modelrefs(self.blockdef, variant):
            # TODO
            if type(modelref) == list:
                modelref = modelref[0]
            if not modelref["model"] in self.models:
                self.models[modelref["model"]] = self._load_modeldef(modelref["model"])
            modelrefs.append(modelref)
        return modelrefs

    def render(self, variant, model, view, projection, rotation=0):
        variant_str = mcmodel.encode_variant(variant)
        if variant_str not in self.variants:
            self.variants[variant_str] = self._load_variant(variant)

        modelrefs = self.variants[variant_str]
        for modelref in modelrefs:
            glmodel = self.models[modelref["model"]]
            glmodel.render(model, view, projection, modelref=modelref, rotation=rotation)

def create_transform_ortho(aspect=1.0, offscreen=False, fake_ortho=True):
    model = np.eye(4, dtype=np.float32)
    if fake_ortho:
        # 0.816479 = 0.5 * sqrt(3) * x = 0.5 * sqrt(2)
        # scale of y-axis to make sides and top of same height
        glm.scale(model, 1.0, 0.816479, 1.0)
        # scale to get block completely into viewport (-1;1)
        glm.scale(model, 1.0 / math.sqrt(2))
    else:
        glm.scale(model, 0.5)
    glm.rotate(model, 45, 0, 1, 0)
    glm.rotate(model, 30, 1, 0, 0)

    view = glm.translation(0, 0, -5)

    projection = glm.ortho(-aspect, aspect, -1, 1, 2.0, 50.0)
    if offscreen:
        glm.scale(projection, 1.0, -1.0, 1.0)

    return model, view, projection

def create_transform_perspective(aspect=1.0, offscreen=False):
    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, 45, 0, 1, 0)
    glm.rotate(model, 25, 1, 0, 0)

    view = glm.translation(0, 0, -5)

    projection = glm.perspective(45.0, aspect, 2.0, 50.0)
    if offscreen:
        glm.scale(projection, 1.0, -1.0, 1.0)

    return model, view, projection

def create_model_transform(rotation=0, phi=0.0):
    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, rotation * 90 + phi, 0, 1, 0)
    return model

