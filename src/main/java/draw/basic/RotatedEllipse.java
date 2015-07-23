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

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.util.Random;
import learning.unsupervised.Cluster;
import statistics.Variance2D;

/**
 * A basic rotated ellipse class for drawing.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class RotatedEllipse {

    public static final Color DEFAULT_COLOR = new Color(255, 0, 0, 127);
    // X coordinate of the center point.
    public double x;
    // Y coordinate of the center point.
    public double y;
    // Ellipse parameters.
    public double a;
    public double b;
    public double angle;
    // Ellipse color.
    private Color color = DEFAULT_COLOR;

    /**
     * Randomizes the ellipse color.
     *
     * @param alpha Integer value of the alpha channel.
     */
    public void randomizeColor(int alpha) {
        // Alpha value given in advance, just randomizing RGB.
        Random randa = new Random();
        int r = randa.nextInt() % 256;
        int g = randa.nextInt() % 256;
        int b = randa.nextInt() % 256;
        color = new Color(r, g, b, alpha);
    }

    /**
     * @param col Color that is to be set for the ellipse drawing.
     */
    public void setColor(Color col) {
        color = col;
    }

    /**
     * @return Color used for ellipse drawing.
     */
    public Color getColor() {
        return color;
    }

    /**
     * The default constructor.
     */
    public RotatedEllipse() {
    }

    /**
     * @param x Center x coordinate.
     * @param y Center y coordinate.
     * @param a Ellipse parameter.
     * @param b Ellipse parameter.
     * @param angle Angle of the rotated ellipse.
     */
    public RotatedEllipse(double x, double y, double a, double b,
            double angle) {
        this.x = x;
        this.y = y;
        this.a = a;
        this.b = b;
        this.angle = angle;
    }

    /**
     * Instantiates a rotated ellipse object from a cluster of SIFT features.
     *
     * @param clust SIFT cluster that is being visualized.
     * @param multiply Boolean flag indicating whether to multiply the lengths
     * of the axes by the confidence scores.
     */
    public RotatedEllipse(Cluster clust, boolean multiply) {
        try {
            Variance2D var = new Variance2D(multiply);
            RotatedEllipse temp = var.findVarianceEllipseForSIFTCluster(clust);
            this.x = temp.x;
            this.y = temp.y;
            this.a = temp.a;
            this.b = temp.b;
            this.angle = temp.angle;
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * Copies the ellipse object.
     *
     * @return Rotated ellipse that is a copy of the current one.
     */
    public RotatedEllipse copy() {
        RotatedEllipse newEllipse = new RotatedEllipse(x, y, a, b, angle);
        return newEllipse;
    }

    /**
     * Draws the ellipse outline on the graphics object.
     *
     * @param g The Graphics2D object to draw the rotated ellipse on.
     */
    public void drawBorderOnGraphics(Graphics2D g) {
        g.setColor(color);
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_ATOP, 1f));
        g.translate(x - a / 2, y - b / 2);
        g.rotate(angle);
        g.draw(new Ellipse2D.Double(0, 0, Math.max((int) a, 4),
                Math.max((int) b, 4)));
        g.rotate(-angle);
        g.translate(a / 2 - x, b / 2 - y);
    }

    /**
     * Draws the ellipse on the graphics object.
     *
     * @param g The Graphics2D object to draw the rotated ellipse on.
     */
    public void drawOnGraphics(Graphics2D g) {
        g.setColor(color);
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_ATOP, 1f));
        g.translate(x - a / 2, y - b / 2);
        g.rotate(angle);
        g.fill(new Ellipse2D.Double(0, 0, Math.max((int) a, 4),
                Math.max((int) b, 4)));
        g.rotate(-angle);
        g.translate(a / 2 - x, b / 2 - y);
    }

    /**
     * Draws the ellipse with a gradient on the graphics object.
     *
     * @param g The Graphics2D object to draw the rotated ellipse on.
     */
    public void drawWithGradient(Graphics2D g) {
        Color endColor = new Color(color.getRed(), color.getGreen(),
                color.getBlue(), 1);
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_ATOP, 1f));
        g.translate(x - a / 2, y - b / 2);
        g.rotate(angle);
        RoundGradientPaint rgp = new RoundGradientPaint(a / 2, b / 2, color,
                new Point2D.Double(0.7 * Math.max(a, b), 0.7 * Math.max(a, b)),
                endColor);
        g.setPaint(rgp);
        g.fill(new Ellipse2D.Double(0, 0, Math.max((int) a, 4),
                Math.max((int) b, 4)));
        g.rotate(-angle);
        g.translate(a / 2 - x, b / 2 - y);
    }

    /**
     * Draws the rotated ellipse on an image.
     *
     * @param bi BufferedImage to draw the rotated ellipse on.
     */
    public void drawOnImage(BufferedImage bi) {
        Graphics2D g = bi.createGraphics();
        drawOnGraphics(g);
    }

    /**
     * Draws the outline of a rotated ellipse on an image.
     *
     * @param bi BufferedImage to draw the rotated ellipse on.
     */
    public void drawBorderOnImage(BufferedImage bi) {
        Graphics2D g = bi.createGraphics();
        drawBorderOnGraphics(g);
    }
}
