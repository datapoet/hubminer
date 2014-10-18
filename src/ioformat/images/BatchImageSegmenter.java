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
package ioformat.images;

import images.mining.segmentation.SRMSegmentation;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import util.CommandLineParser;

/**
 * This class implements methods for batch segmenting a directory of images in
 * JPG format.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchImageSegmenter {

    private BufferedImage currentImage;
    private BufferedImage segmentedImage;
    private String name;
    private File inDir;
    private File outImgDir;
    private File outRepDir;

    /**
     * Initialization.
     *
     * @param inDir File that is the directory containing the images to be
     * segmented.
     * @param outImgDir File that is the directory that will contain the
     * segmented images.
     * @param outRepDir File that is the directory that will contain the
     * segmentation representations for the segmented images.
     */
    public BatchImageSegmenter(File inDir, File outImgDir, File outRepDir) {
        this.inDir = inDir;
        this.outImgDir = outImgDir;
        this.outRepDir = outRepDir;
    }

    /**
     * Perform the batch image segmentation.
     *
     * @throws Exception
     */
    public void segment() throws Exception {
        File[] children = inDir.listFiles();
        SRMSegmentation seg;
        int index;
        for (int i = 0; i < children.length; i++) {
            if ((children[i].getPath().endsWith(".jpg"))
                    || (children[i].getPath().endsWith(".JPG"))
                    || (children[i].getPath().endsWith(".jpeg"))) {
                currentImage = ImageIO.read(children[i]);
                index = children[i].getName().lastIndexOf(".");
                if (index > 0) {
                    name = children[i].getName().substring(0, index);
                }
                seg = new SRMSegmentation(currentImage);
                seg.segment();
                // Save the results.
                Image img = seg.getSegmentedImage();
                segmentedImage = new BufferedImage(img.getWidth(null),
                        img.getHeight(null), BufferedImage.TYPE_INT_RGB);
                segmentedImage.getGraphics().drawImage(img, 0, 0, null);
                ImageIO.write(segmentedImage, "JPG", new File(outImgDir,
                        name + ".jpg"));
                SegmentationIO segIO = new SegmentationIO();
                segIO.write(seg, new File(outRepDir, name + ".txt"));
            }
        }
    }

    /**
     * This script executes batch image SRM segmentation.
     *
     * @param args String[] of command line parameters including the input and
     * output directories, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inImages", "Path to the input image directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outImages", "Path to the output image directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outSegmentations", "Path to output the segmentations "
                + "to.", CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File inImageDir = new File((String) clp.getParamValues("-inImages").
                get(0));
        File outImageDir = new File((String) clp.getParamValues("-outImages").
                get(0));
        File outSegDir = new File((String) clp.getParamValues(
                "-outSegmentations").get(0));
        BatchImageSegmenter bis = new BatchImageSegmenter(inImageDir,
                outImageDir, outSegDir);
        bis.segment();
    }
}
