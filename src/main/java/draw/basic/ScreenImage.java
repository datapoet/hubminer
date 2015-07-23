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

import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.util.Arrays;
import java.util.List;
import javax.imageio.*;
import javax.swing.*;

/**
 * This is a convenience class to create and optionally save to a file a
 * BufferedImage of an area shown on the screen. It covers several different
 * scenarios, so the image can be created of:
 *
 * a) an entire component b) a region of the component c) the entire desktop d)
 * a region of the desktop
 *
 * This class can also be used to create images of components not displayed on a
 * GUI. The only foolproof way to get an image in such cases is to make sure the
 * component has been added to a realized window with code something like the
 * following:
 *
 * JFrame frame = new JFrame(); frame.setContentPane( someComponent );
 * frame.pack(); ScreenImage.createImage( someComponent );
 *
 * @author This code has been taken from the following web sources:
 * https://www.wuestkamp.com/2012/02/java-save-component-to-image-make-screen-shot-of-component/
 * http://tips4java.wordpress.com/2008/10/13/screen-image/
 */
public class ScreenImage {

    private static List<String> imageTypes = Arrays.asList(
            ImageIO.getWriterFileSuffixes());

    /**
     * Create a BufferedImage for Swing components. The entire component will be
     * captured to an image.
     *
     * @param component Swing component to create the image from.
     * @return	image The image for the given region.
     */
    public static BufferedImage createImage(JComponent component) {
        Dimension dim = component.getSize();
        if (dim.width == 0 || dim.height == 0) {
            dim = component.getPreferredSize();
            component.setSize(dim);
        }
        Rectangle region = new Rectangle(0, 0, dim.width, dim.height);
        return ScreenImage.createImage(component, region);
    }

    /**
     * Create a BufferedImage for Swing components. All or part of the component
     * can be captured to an image.
     *
     * @param component Swing component to create the image from.
     * @param region The region of the component to be captured to an image.
     * @return	image The image for the given region.
     */
    public static BufferedImage createImage(JComponent component,
            Rectangle region) {
        if (!component.isDisplayable()) {
            Dimension dim = component.getSize();
            if (dim.width == 0 || dim.height == 0) {
                dim = component.getPreferredSize();
                component.setSize(dim);
            }
            layoutComponent(component);
        }
        BufferedImage image = new BufferedImage(region.width, region.height,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        // Paint a background for non-opaque components.
        if (!component.isOpaque()) {
            g2d.setColor(component.getBackground());
            g2d.fillRect(region.x, region.y, region.width, region.height);
        }
        g2d.translate(-region.x, -region.y);
        component.paint(g2d);
        g2d.dispose();
        return image;
    }

    /**
     * Convenience method to create a BufferedImage of the desktop.
     *
     * @param fileName Name of file to be created or null.
     * @return	image The image for the given region.
     * @exception AWTException
     * @exception IOException
     */
    public static BufferedImage createDesktopImage() throws AWTException,
            IOException {
        Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
        Rectangle region = new Rectangle(0, 0, dim.width, dim.height);
        return ScreenImage.createImage(region);
    }

    /**
     * Create a BufferedImage for AWT components.
     *
     * @param component AWT component to create image from
     * @return	image the image for the given region
     * @exception AWTException see Robot class constructors
     */
    public static BufferedImage createImage(Component component)
            throws AWTException {
        Point p = new Point(0, 0);
        SwingUtilities.convertPointToScreen(p, component);
        Rectangle region = component.getBounds();
        region.x = p.x;
        region.y = p.y;
        return ScreenImage.createImage(region);
    }

    /**
     * Create a BufferedImage from a rectangular region on the screen. This will
     * include Swing components JFrame, JDialog and JWindow which all extend
     * from Component, not JComponent.
     *
     * @param Region region on the screen to create image from
     * @return	Image the image for the given region
     * @exception AWTException see Robot class constructors
     */
    public static BufferedImage createImage(Rectangle region)
            throws AWTException {
        BufferedImage image = new Robot().createScreenCapture(region);
        return image;
    }

    /**
     * Write a BufferedImage to a File.
     *
     * @param Image image to be written.
     * @param FileName name of file to be created.
     * @exception IOException if an error occurs during writing.
     */
    public static void writeImage(BufferedImage image, String fileName)
            throws IOException {
        if (fileName == null) {
            return;
        }
        int offset = fileName.lastIndexOf(".");
        if (offset == -1) {
            String message = "file type was not specified";
            throw new IOException(message);
        }
        String fileType = fileName.substring(offset + 1);
        if (imageTypes.contains(fileType)) {
            ImageIO.write(image, fileType, new File(fileName));
        } else {
            String message = "unknown writer file type (" + fileType + ")";
            throw new IOException(message);
        }
    }

    /**
     * A recursive layout call on the component.
     *
     * @param component Component object.
     */
    static void layoutComponent(Component component) {
        synchronized (component.getTreeLock()) {
            component.doLayout();
            if (component instanceof Container) {
                for (Component child :
                        ((Container) component).getComponents()) {
                    layoutComponent(child);
                }
            }
        }
    }
}
