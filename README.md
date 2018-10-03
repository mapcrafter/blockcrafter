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

This makes Blockcrafter export the Minecraft 1.13.1 block images of isometric render view with texture size 12px and rotation 0 (top-left) to the directory `blocks/`. You can also export block images of multiple render views / texture sizes / rotations by just passing multiple `-v`/`-t`/`-r` arguments. Blockcrafter will then export block images for all configurations render views `x` texture sizes `x` rotations. See below for a reference of all options.

### Command line options ###

TODO
