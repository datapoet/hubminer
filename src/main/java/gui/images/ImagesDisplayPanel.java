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
package gui.images;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import javax.swing.JPanel;

/**
 * This panel is used for displaying an image dataset. The background image is
 * set to the desired landscape and the image coordinates are provided.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImagesDisplayPanel extends JPanel {

    // Backround image.
    protected Image bckgImage = null;
    // Image dataset.
    protected ArrayList<ImagePanelWithClass> imageSet = null;
    // Image bounding rectangles.
    protected ArrayList<Rectangle2D> imageRects = null;

    /**
     * Initialization.
     */
    public ImagesDisplayPanel() {
        this.setPreferredSize(new Dimension(2000, 2000));
        this.setMaximumSize(new Dimension(2000, 2000));
        this.setMinimumSize(new Dimension(2000, 2000));
        this.setOpaque(true);
        setLayout(null);
    }

    /**
     * @param bckgImage Image that is to be used as background.
     */
    public void setBackgroundImage(Image bckgImage) {
        this.bckgImage = bckgImage;
        revalidate();
        repaint();
    }

    /**
     * @return Image that is used as background.
     */
    public Image getBackgroundImage() {
        return bckgImage;
    }

    /**
     * @param imageSet ArrayList<ImagePanelWithClass> of images that are to be
     * displayed. The total image dataset can be larger.
     * @param imageRects ArrayList<Rectangle2D> of bounding rectangles for the
     * images that are to be displayed.
     */
    public void setImageSet(ArrayList<ImagePanelWithClass> imageSet,
            ArrayList<Rectangle2D> imageRects) {
        this.removeAll();
        for (int index = 0; index < imageSet.size(); index++) {
            this.add(imageSet.get(index));
            imageSet.get(index).setBounds((int) imageRects.get(index).getX(),
                    (int) imageRects.get(index).getY(),
                    (int) imageRects.get(index).getWidth(),
                    (int) imageRects.get(index).getHeight());
        }
        this.imageSet = imageSet;
        this.imageRects = imageRects;
        revalidate();
        repaint();
    }

    /**
     * @param background Image that is to be used as background.
     * @param imageSet ArrayList<ImagePanelWithClass> of images that are to be
     * displayed. The total image dataset can be larger.
     * @param imageRects ArrayList<Rectangle2D> of bounding rectangles for the
     * images that are to be displayed.
     */
    public void setImageSet(Image background,
            ArrayList<ImagePanelWithClass> imageSet,
            ArrayList<Rectangle2D> imageRects) {
        bckgImage = background;
        setBackgroundImage(bckgImage);
        setImageSet(imageSet, imageRects);
    }

    public ArrayList<ImagePanelWithClass> getImageSet() {
        return imageSet;
    }

    @Override
    public void update(Graphics g) {
        if (bckgImage != null) {
            g.drawImage(bckgImage, 0, 0, this.getSize().width,
                    this.getSize().height, this);
        } else {
            Color c = g.getColor();
            g.setColor(Color.GRAY);
            g.fillRect(0, 0, this.getWidth(), this.getHeight());
            g.setColor(c);
        }
    }

    @Override
    public void paintComponent(Graphics g) {
        update(g);
    }
}
