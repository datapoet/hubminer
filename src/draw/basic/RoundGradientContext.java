/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package draw.basic;

import java.awt.Color;
import java.awt.PaintContext;
import java.awt.geom.Point2D;
import java.awt.image.ColorModel;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;

/**
 * This class implements the paint context for round gradients.
 *
 * @author This code is taken from the following example on the web:
 * http://www.java2s.com/Code/Java/2D-Graphics-GUI/RoundGradientPaintFilldemo.htm
 */
public class RoundGradientContext implements PaintContext {

    private Point2D middlePoint;
    private double radius = 0;
    private Color firstColor, secondColor;

    /**
     *
     * @param middlePoint Point2D object that is the middle point.
     * @param firstColor Color that is the color in the middle.
     * @param radius Double value corresponding to the radius of the circle.
     * @param secondColor Color that is the color at the border, towards which
     * the middle color is slowly changed by the gradient.
     */
    public RoundGradientContext(Point2D middlePoint, Color firstColor,
            double radius, Color secondColor) {
        this.middlePoint = middlePoint;
        this.firstColor = firstColor;
        this.radius = radius;
        this.secondColor = secondColor;
    }

    @Override
    public void dispose() {
    }

    @Override
    public ColorModel getColorModel() {
        return ColorModel.getRGBdefault();
    }

    @Override
    public Raster getRaster(int x, int y, int width, int height) {
        WritableRaster writableRaster = getColorModel().
                createCompatibleWritableRaster(width, height);
        int[] data = new int[width * height * 4];
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                double distance = middlePoint.distance(x + i, y + j);
                double ratio = distance / radius;
                if (ratio > 1.0) {
                    ratio = 1.0;
                }
                int value = (j * width + i) * 4;
                data[value + 0] = (int) (firstColor.getRed()
                        + ratio * (secondColor.getRed() - firstColor.getRed()));
                data[value + 1] = (int) (firstColor.getGreen() + ratio
                        * (secondColor.getGreen() - firstColor.getGreen()));
                data[value + 2] = (int) (firstColor.getBlue()
                        + ratio * (secondColor.getBlue() -
                        firstColor.getBlue()));
                data[value + 3] = (int) (firstColor.getAlpha() + ratio
                        * (secondColor.getAlpha() - firstColor.getAlpha()));
            }
        }
        writableRaster.setPixels(0, 0, width, height, data);
        return writableRaster;
    }
}
