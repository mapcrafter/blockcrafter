# Blockcrafter #

by Moritz Hilscher

Blockcrafter is a tool to render Minecraft block images for [Mapcrafter](https://github.com/mapcrafter/mapcrafter).

## Installation ##

Blockcrafter requires Python 3 and the packages numpy, vispy and Pillow.

### From source ###

Just run `pip3 install .` in the directory.

If you want to work with Blockcrafter, run `pip3 install --user -e .`. This installs Blockcrafter locally but you can still edit the cloned sources. Make sure that `~/.local/bin` is in your path.

### Docker ###

Just pull the docker image `mapcrafter/blockcrafter`.

## Usage ##

Blockcrafter provides the script `blockcrafter-export` to render Minecraft block images and export them for Mapcrafter.

Installation with pip, run:

`blockcrafter-export <arguments>`

With docker, you have to run:

`docker run --rm -v $(pwd):/wd -w /wd -e LOCAL_USER_ID=$(id -u $(whoami)) blockcrafter <arguments>`

(`--rm` makes docker delete the container after running it. `-v $(pwd):/wd` mounts the current directory into the container, `-w /wd` sets the working directory inside the container, and also your user ID is passed to the container so it's like Blockcrafter is running in your current directory as your user.)

### Example ###

You need to get a Minecraft client jar file of version 1.13 at least. Then you can run `blockcrafter-export` (or the equivalent docker command):

`blockcrafter-export -a 1.13.1.jar -o blocks -v isometric -t 12 -r 0`

This makes Blockcrafter export the Minecraft 1.13.1 block images of isometric render view with texture size 12px and rotation 0 (top-left) to the directory `blocks/`. You can also export block images of multiple render views / texture sizes / rotations by just passing multiple `-v`/`-t`/`-r` arguments. Blockcrafter will then export block images for all configurations `render views x texture sizes x rotations`. See below for a reference of all options.

### Command line options ###

#### Main options you will be using ####

`-a <asset>, --asset=<asset>` **required**: Input Minecraft asset. Can be path to Minecraft client jar, to a resource pack zip, or to a directory of any of these unpacked. If you want to use a custom texture pack, it's important that you pass the Minecraft assets first and then the texture pack, for example `-a 1.13.1.jar -a texturepack.zip`.

`-o <directory>, --output-dir=<directory>` **required**: Output directory where Blockcrafter should export block images to. 

`-v <view>, --view=<view>`: Render view to export block images for, must be one of `isometric`, `topdown`, or `side`. You can use this option multiple times to export block images for multiple views, for example `-v isometric -v topdown`. If you don't specify this option, block images are exported for all views.

`-t <size>, --texture-size=<size>`: Texture size to export block images for. Like `--view`, you can use this option multiple times. Defaults to `-t 12 -t 16`.

`-r <rotation>, --rotation=<rotation>`: Rotation to export block images for. Like `--view` and `--texture-size`, you can use this option multiple times. Defaults to `-r 0 -r 1 -r 2 -r 3`. Rotation indices stand for rotations top-left, top-right, bottom-right, and bottom-left.

#### Additional options ####

`--osmesa`: Use osmesa software OpenGL rendering. This is enabled per default when running Blockcrafter in the docker container.

`-b <wildcard>, --blocks=<wildcard>`: You can specify a wildcard to render only certain blocks, useful for debugging certain block images. The wildcard is applied for blockstate names such as `minecraft:oak-fence`. You can use the option multiple times to match multiple groups of blockstates.

`--no-render`: Don't render any block image, just write out the `.txt` block info file again.
