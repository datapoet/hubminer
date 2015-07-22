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
import java.awt.Paint;
import java.awt.PaintContext;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.ColorModel;

/**
 * This class implements the paint for round gradients.
 *
 * @author This code is taken from the following example on the web:
 * http://www.java2s.com/Code/Java/2D-Graphics-GUI/RoundGradientPaintFilldemo.htm
 */
class RoundGradientPaint implements Paint {

    private Point2D middlePoint;
    private Point2D radiusPoint;
    private Color firstColor, secondColor;

    /**
     * Initialization.
     *
     * @param x The X coordinate of the middle point.
     * @param y The Y coordinate of the middle point.
     * @param firstColor Color that is the color in the middle.
     * @param radiusPoint Point defining the radius.
     * @param secondColor Color that is the color at the border, towards which
     * the middle color is slowly changed by the gradient.
     */
    public RoundGradientPaint(double x, double y, Color firstColor,
            Point2D radiusPoint, Color secondColor) {
        middlePoint = new Point2D.Double(x, y);
        this.firstColor = firstColor;
        this.radiusPoint = radiusPoint;
        this.secondColor = secondColor;
    }

    @Override
    public PaintContext createContext(ColorModel colorModel, Rectangle rect,
            Rectangle2D rect2D, AffineTransform affineTransform,
            RenderingHints rh) {
        Point2D transPoint = affineTransform.transform(middlePoint, null);
        Point2D transRadius = affineTransform.deltaTransform(radiusPoint, null);
        double radius = transRadius.distance(0, 0);
        return new RoundGradientContext(transPoint, firstColor, radius,
                secondColor);
    }

    /**
     * Creates the corresponding paint context.
     *
     * @param colorModel ColorModel object.
     * @param affineTransform AffineTransform object defining the
     * transformation.
     * @return
     */
    public PaintContext createContext(ColorModel colorModel,
            AffineTransform affineTransform) {
        Point2D transPoint = affineTransform.transform(middlePoint, null);
        Point2D transRadius = affineTransform.deltaTransform(radiusPoint, null);
        double radius = transRadius.distance(0, 0);
        return new RoundGradientContext(transPoint, firstColor, radius,
                secondColor);
    }

    @Override
    public int getTransparency() {
        int alphaFirst = firstColor.getAlpha();
        int alphaSecond = secondColor.getAlpha();
        return (((alphaFirst & alphaSecond) == 0xff) ? OPAQUE : TRANSLUCENT);
    }
}
