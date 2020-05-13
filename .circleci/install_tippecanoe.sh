sudo apt-get update -y && sudo apt-get install libgdal-dev
git clone git@github.com:mapbox/tippecanoe.git
cd tippecanoe
sudo make -j
sudo make install