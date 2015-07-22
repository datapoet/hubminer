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
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 * This class is an image-holding panel.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImagePanel extends JPanel {

    // Image object that is to be shown.
    protected Image image = null;
    // In case no image is provided, there is a name that can be shown.
    private String objectName = null;

    /**
     * Default constructor.
     */
    public ImagePanel() {
    }

    /**
     * @param newImage Image object to be shown.
     */
    public void setImage(Image newImage) {
        image = newImage;
        validate();
        repaint();
    }

    /**
     * @return Image object to be shown.
     */
    public Image getImage() {
        return image;
    }

    /**
     * @param objectName String that is the object name. It is shown in absence
     * of an image.
     */
    public void setObjectName(String objectName) {
        JLabel lab = new JLabel(objectName);
        lab.setFont(new Font("Verdana", 1, 8));
        this.setLayout(new BorderLayout());
        this.add(lab, BorderLayout.CENTER);
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
            g.drawImage(image, 0, 0, this.getSize().width,
                    this.getSize().height, this);
        } else if (objectName != null) {
            super.paint(g);
        } else {
            // If no information is provided, just paint a rectangle.
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
