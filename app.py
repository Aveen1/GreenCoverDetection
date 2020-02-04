
from flask import Flask, render_template, make_response

import ee
import numpy as np
from skimage.segmentation import slic, mark_boundaries
import cv2




import time

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template("index1.html")


@app.route('/map')
def map():
    return render_template("test.html")

@app.route('/result')
def result():
    return render_template("result.html")

@app.route('/code')
def code():
    # init the ee object
    ee.Initialize()

    with open("/Users/admin/Downloads/scene.geojson", "r") as file:
        geojson = file.readlines()
        geojson = eval(geojson[0])
        geometry = geojson["geometry"]
        cords = geometry["coordinates"]
        polygon = cords[0]
    print(polygon)

    def stretch_8bit(bands, lower_percent=2, higher_percent=98):
        out = np.zeros_like(bands)
        for i in range(3):
            a = 0  # np.min(band)
            b = 255  # np.max(band)
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t
        return out.astype(np.uint8)

        # Define the area

    area = ee.Geometry.Polygon(polygon)

    # define the image

    coll = ee.ImageCollection("COPERNICUS/S2_SR")
    image_area = coll.filterBounds(area)
    img = image_area.median()

    timedate = img.get('GENERATION_TIME').getInfo()

    # get the lat lon and add the ndvi

    bands = ["B4", "B3", "B2"]
    band_outputs = {}
    # red
    for band in bands:

        image = img.select(band).rename(["temp"])

        latlon = ee.Image.pixelLonLat().addBands(image)

        latlon = latlon.reduceRegion(
            reducer=ee.Reducer.toList(),
            geometry=area,
            maxPixels=1e8,
            scale=10)

        data = np.array((ee.Array(latlon.get("temp")).getInfo()))
        lats = np.array((ee.Array(latlon.get("latitude")).getInfo()))
        lons = np.array((ee.Array(latlon.get("longitude")).getInfo()))

        #get the unique coordinates
        uniqueLats = np.unique(lats)
        uniqueLons = np.unique(lons)

        #get number of columns and rows from coordinates
        ncols = len(uniqueLons)
        nrows = len(uniqueLats)

        #determine pixelsizes
        ys = uniqueLats[1] - uniqueLats[0]
        xs = uniqueLons[1] - uniqueLons[0]

        #create an array with dimensions of image
        arr = np.zeros([nrows, ncols], np.float32)  # -9999

        # fill the array with values
        counter = 0
        for y in range(0, len(arr), 1):
            for x in range(0, len(arr[0]), 1):
                if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats) - 1:
                    counter += 1
                    arr[len(uniqueLats) - 1 - y, x] = data[counter]  # we start from lower left corner
        band_outputs[band] = arr

    r = np.expand_dims(band_outputs["B4"], -1).astype("float32")
    g = np.expand_dims(band_outputs["B3"], -1).astype("float32")
    b = np.expand_dims(band_outputs["B2"], -1).astype("float32")
    rgb = np.concatenate((r, g, b), axis=2) / 3000

    coll = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA")
    image_area = coll.filterBounds(area)
    img = image_area.median()

    RED = img.select("B4")
    NIR = img.select("B5")
    NDVI = ee.Image(img.subtract(RED).divide(NIR.add(RED)))

    #get the lat lon and add the ndvi
    latlon = ee.Image.pixelLonLat().addBands(NDVI)

    #apply reducer to list
    latlon = latlon.reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=area,
        maxPixels=1e8,
        scale=10)

    data = np.array((ee.Array(latlon.get("B5")).getInfo()))
    lats = np.array((ee.Array(latlon.get("latitude")).getInfo()))
    lons = np.array((ee.Array(latlon.get("longitude")).getInfo()))
    print(data.shape)
    #get the unique coordinates
    uniqueLats = np.unique(lats)
    uniqueLons = np.unique(lons)

    #get number of columns and rows from coordinates
    ncols = len(uniqueLons)
    nrows = len(uniqueLats)

    #determine pixelsizes
    ys = uniqueLats[1] - uniqueLats[0]
    xs = uniqueLons[1] - uniqueLons[0]

    #create an array with dimensions of image
    arr = np.zeros([nrows, ncols], np.float32)  # -9999

    #fill the array with values
    counter = 0
    for y in range(0, len(arr), 1):
        for x in range(0, len(arr[0]), 1):
            if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats) - 1:
                counter += 1
                arr[len(uniqueLats) - 1 - y, x] = data[counter]  # we start from lower left corner

    #MASKING
    ndvi = (arr ** 2).copy()
    ndvi[ndvi < 0.05] = 0
    ndvi[ndvi > 0.05] = 1
    blue = rgb[:, :, 0]
    red = rgb[:, :, 2]
    blue[ndvi == 1] -= 0.5
    red[ndvi == 1] -= 0.5
    output = rgb.copy()
    output[:, :, 0] = blue
    output[:, :, 2] = red
    output *= 255
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    cv2.imwrite("static/v.jpg", output)

    #CALCULATIONS
    total_img = np.sum(rgb, axis=-1)
    total = len(total_img[total_img != 0])
    green = len(ndvi[ndvi != 0])
    build = total-green
    percent = (green / total) * 100

    if percent < 40:
        p1="Plantation less than 40% is not considered as optimal so kindly plant more trees."
    else:
        p1="Plantation is above 40%, you can plant more if you need or help others to plant trees."


    return render_template("result.html", b=total, c=green, d=build, e=percent, f=p1)





if __name__ == '__main__':
    app.run(debug=True)













