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

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 * This class is an image-holding panel that also displays the class-affiliation
 * information as a colored frame around the image.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImagePanelWithClass extends JPanel {

    // An array of colors that are used for different classes.
    private Color[] classColors;
    // The index of the class of this image.
    int classIndex;
    // Image index in the embedding image dataset.
    int imageIndex;
    // Object name that is used for display in absence of an image.
    private String objectName = null;
    // Image that is to be displayed.
    protected Image image = null;

    /**
     * Initialization.
     *
     * @param classColors Color[] of class colors.
     */
    public ImagePanelWithClass(Color[] classColors) {
        this.classColors = classColors;
        setPreferredSize(new Dimension(80, 80));
        setMaximumSize(new Dimension(80, 80));
        setMinimumSize(new Dimension(80, 80));
    }

    /**
     * Sets the image to be shown.
     *
     * @param newImage Image object that is to be shown.
     * @param classIndex Integer that is the class index.
     * @param imageIndex Integer that is the image index.
     */
    public void setImage(Image newImage, int classIndex, int imageIndex) {
        image = newImage.getScaledInstance((int) (80 * 0.8), (int) (80 * 0.8),
                Image.SCALE_DEFAULT);
        this.classIndex = classIndex;
        this.imageIndex = imageIndex;
        validate();
        repaint();
    }

    /**
     * @return Image object that is to be shown.
     */
    public Image getImage() {
        return image;
    }

    /**
     * @return Integer that is the image index in the embedding image dataset.
     */
    public int getImageIndex() {
        return imageIndex;
    }

    /**
     * @param objectName String that is the object name. It is shown in absence
     * of an image.
     * @param classIndex Integer that is the class index.
     * @param imageIndex Integer that is the image index.
     */
    public void setObjectName(String objectName, int classIndex,
            int objectIndex) {
        JLabel lab = new JLabel(objectName);
        lab.setFont(new Font("Verdana", 1, 8));
        this.setLayout(new BorderLayout());
        this.add(lab, BorderLayout.CENTER);
        this.classIndex = classIndex;
        this.imageIndex = objectIndex;
    }

    /**
     * @return String that is the object name. It is shown in absence of an
     * image.
     */
    public String getObjectName() {
        return objectName;
    }

    @Override
    public void update(Graphics g) {
        if (image != null) {
            g.setColor(classColors[classIndex]);
            g.fillRect(0, 0, this.getWidth(), this.getHeight());
            // The image is drawn within the colored frame.
            g.drawImage(image, 0, 0, (int) (this.getSize().width * 0.8),
                    (int) (this.getSize().height * 0.8), this);
        } else if (objectName != null) {
            // The object name is used instead. The named label is shown.
            super.paint(g);
            g.setColor(classColors[classIndex]);
            g.fillRect((int) (this.getSize().width * 0.8), 0, this.getWidth(),
                    this.getHeight());
            g.fillRect(0, (int) (this.getSize().height * 0.8), this.getWidth(),
                    this.getHeight());
        } else {
            Color c = g.getColor();
            g.setColor(Color.GRAY);
            g.fillRect(0, 0, this.getWidth(), this.getHeight());
            g.setColor(c);
        }
    }

    @Override
    public void paint(Graphics g) {
        update(g);
    }
}
