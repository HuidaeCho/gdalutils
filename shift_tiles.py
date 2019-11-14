#!/usr/bin/env python3
################################################################################
# Name:    shift_tiles.py
# Purpose: This module provides useful functions that can shift and snap
#          small raster tiles to a bigger reference raster.
# Author:  Huidae Cho
# Since:   November 14, 2019
#
# Copyright (C) 2019, Huidae Cho <https://idea.isnew.info>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
################################################################################

import gdal
import numpy as np

W = 1<<0
N = 1<<1
E = 1<<2
S = 1<<3

def calc_xy(gt, col, row):
    x = gt[0] + col*gt[1] + row*gt[2];
    y = gt[3] + col*gt[4] + row*gt[5];
    return np.asarray((x, y))

def calc_colrow(gt, x, y):
    col = (x*gt[5] - y*gt[2] - gt[0]*gt[5] + gt[2]*gt[3]) / (gt[1]*gt[5] - gt[2]*gt[4])
    row = (x*gt[4] - y*gt[1] - gt[0]*gt[4] + gt[1]*gt[3]) / (gt[2]*gt[4] - gt[1]*gt[5])
    return np.asarray((col, row))

def calc_shift_skip(refband, refgt, outa, outgt, skipw=0, skipn=0, skipe=0, skips=0):
    ncols = outa.shape[1]
    nrows = outa.shape[0]
    if skipw + skipn + skipe + skips:
        outa = outa[skipn:nrows-skips, skipw:ncols-skipe]
        ncols = outa.shape[1]
        nrows = outa.shape[0]

    outnwxy = calc_xy(outgt, skipw, skipn)
    outnwcr = calc_colrow(refgt, outnwxy[0], outnwxy[1])
    outnwcrf = [int(x) for x in np.floor(outnwcr)]

    for coff in range(2):
        for roff in range(2):
            c = outnwcrf[0] + coff
            r = outnwcrf[1] + roff
            refouta = refband.ReadAsArray(c, r, ncols, nrows)
            diffsum = np.sum(np.where(outa < 255, outa - 1 - refouta, 0))
            if diffsum == 0:
                # found shift
                refcrxy = calc_xy(refgt, c, r)
                return refcrxy - outnwxy
    # overlapping regions are dominated by neighbor tiles
    return None

def calc_shift(refband, refgt, outband, outgt):
    outa = outband.ReadAsArray()
    shift = calc_shift_skip(refband, refgt, outa, outgt)
    if shift is not None:
        # dominating tile
        return shift, 0

    # overlapping regions are dominated by neighbor tiles
    ncols = outa.shape[1]
    nrows = outa.shape[0]

    skipwe = int(ncols/3)
    skipns = int(nrows/3)

    # test west
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipw=skipwe)
    if shift is not None:
        # west side is overwritten
        return shift, W

    # test north
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipn=skipns)
    if shift is not None:
        # north side is overwritten
        return shift, N

    # test east
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipe=skipwe)
    if shift is not None:
        # east side is overwritten
        return shift, E

    # test south
    shift = calc_shift_skip(refband, refgt, outa, outgt, skips=skipns)
    if shift is not None:
        # south side is overwritten
        return shift, S

    # test west, north
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipw=skipwe, skipn=skipns)
    if shift is not None:
        # west, north sides are overwritten
        return shift, W|N

    # test west, east
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipw=skipwe, skipe=skipwe)
    if shift is not None:
        # west, east sides are overwritten
        return shift, W|E

    # test west, south
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipw=skipwe, skips=skipns)
    if shift is not None:
        # west, south sides are overwritten
        return shift, W|S

    # test north, east
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipn=skipns, skipe=skipwe)
    if shift is not None:
        # north, east sides are overwritten
        return shift, N|E

    # test north, south
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipn=skipns, skips=skipns)
    if shift is not None:
        # north, south sides are overwritten
        return shift, N|S

    # test east, south
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipe=skipwe, skips=skipns)
    if shift is not None:
        # east, south sides are overwritten
        return shift, E|S

    # test west, north, east
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipw=skipwe, skipn=skipns, skipe=skipwe)
    if shift is not None:
        # west, north, east sides are overwritten
        return shift, W|N|E

    # test west, north, south
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipw=skipwe, skipn=skipns, skips=skipns)
    if shift is not None:
        # west, north, south sides are overwritten
        return shift, W|N|S

    # test west, east, south
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipw=skipwe, skipe=skipwe, skips=skipns)
    if shift is not None:
        # west, north, south sides are overwritten
        return shift, W|E|S

    # test north, east, south
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipn=skipns, skipe=skipwe, skips=skipns)
    if shift is not None:
        # north, east, south sides are overwritten
        return shift, N|E|S

    # test all sides
    shift = calc_shift_skip(refband, refgt, outa, outgt, skipw=skipwe, skipn=skipns, skipe=skipwe, skips=skipns)
    if shift is not None:
        # all sides are overwritten
        return shift, W|N|E|S

    # failed to calculate shift
    return None, None

def shift(refband, refgt, outfile, newfile):
    outrast = gdal.Open(outfile)
    outband = outrast.GetRasterBand(1)
    outgt = outrast.GetGeoTransform()
    shift, overlap = calc_shift(refband, refgt, outband, outgt)

    if shift is None:
        print('Failed to shift')
        return

    print(shift, overlap)

    outgt = list(outgt)
    outgt[0] += shift[0]
    outgt[3] += shift[1]
    driver = outrast.GetDriver()
    newrast = driver.Create(newfile, outband.XSize, outband.YSize, 1, gdal.GDT_Byte)
    newrast.SetGeoTransform(outgt)
    newband = newrast.GetRasterBand(1)
    newband.SetNoDataValue(outband.GetNoDataValue())
    newband.WriteArray(outband.ReadAsArray())
    newrast.SetProjection(outrast.GetProjection())
    newrast.FlushCache()

def print_shift(refband, refgt, outfile):
    outrast = gdal.Open(outfile)
    outband = outrast.GetRasterBand(1)
    outgt = outrast.GetGeoTransform()
    shift, overlap = calc_shift(refband, refgt, outband, outgt)

    if shift is None:
        print('Failed to calculate shift')
    else:
        print(shift, overlap)
