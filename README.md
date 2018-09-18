# Blockcrafter #

by Moritz Hilscher

Blockcrafter is a tool to render Minecraft block images for [Mapcrafter](https://github.com/mapcrafter/mapcrafter).

## Installation ##

Blockcrafter requires Python 3 and the packages numpy, vispy and Pillow.

### From source ###

Just run `pip install .` in the directory.

If you want to work with Blockcrafter, run `pip install --user -e .`. This installs Blockcrafter locally but you can still edit the cloned sources. Make sure that `~/.local/bin` is in your path.

### Docker ###

Just pull the docker image `mapcrafter/blockcrafter`.

## Usage ##

Blockcrafter has two scripts:

- `blockcrafter-export`: Main tool, renders block images for Mapcrafter. 
- `blockcrafter-visualize`: Development tool, can visualize single blocks.

TODO
