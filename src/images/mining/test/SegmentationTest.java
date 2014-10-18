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
package images.mining.test;

import images.mining.segmentation.SRMSegmentation;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

/**
 * Tests how the SRM image segmentation works.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SegmentationTest {

    /**
     * This executes the script.
     * @param args Two parameters, as specified.
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("arg0: Path to the input image file.");
            System.out.println("arg1: Path to the output file for the segmented"
                    + " image.");
            return;
        }
        BufferedImage bi = ImageIO.read(new File(args[0]));
        SRMSegmentation seg = new SRMSegmentation(bi);
        seg.segment();
        BufferedImage br;
        Image img = seg.getSegmentedImage();
        br = new BufferedImage(img.getWidth(null), img.getHeight(null),
                BufferedImage.TYPE_INT_RGB);
        br.getGraphics().drawImage(img, 0, 0, null);
        ImageIO.write(br, "jpg", new File(args[1]));
    }
}
