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

import data.representation.images.sift.LFeatRepresentation;
import data.representation.images.sift.LFeatVector;
import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

/**
 * This class handles saving and loading keypoints and descriptors in the OpenCV
 * data format. OpenCV is a widely used image processing library used for
 * feature extraction.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class OpenCVFeatureIO {
    
    /**
     * Load the local feature representation from the keypoint and descriptor
     * files in the OpenCV format.
     * 
     * @param keypointFile File that holds the keypoint specification for the
     * image.
     * @param descriptorFile File that holds the descriptor value for the image.
     * @return LFeatRepresentation loaded from the files.
     * @throws IOException 
     */
    public static LFeatRepresentation loadImageRepresentation(File keypointFile,
            File descriptorFile) throws IOException {
        if (keypointFile == null) {
            throw new NullPointerException("Null keypoint file.");
        }
        if (descriptorFile == null) {
            throw new NullPointerException("Null descriptor file.");
        }
        if (!keypointFile.exists()) {
            throw new IOException("Keypoint file " + keypointFile.getPath() +
                    " does not exist.");
        }
        if (!descriptorFile.exists()) {
            throw new IOException("Descriptor file " +
                    descriptorFile.getPath() + " does not exist.");
        }
        // First take a sneek peak into the descriptor file to get the
        // descriptor length.
        int descriptorLength;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(descriptorFile)))) {
            String s = br.readLine();
            String[] lineItems = s.split(",");
            descriptorLength = lineItems.length;
        }
        LFeatRepresentation imageRepresentation = new LFeatRepresentation(
                descriptorLength);
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(descriptorFile)))) {
            String s = br.readLine();
            String[] lineItems;
            while (s != null) {
                lineItems = s.split(",");
                LFeatVector vect = new LFeatVector(imageRepresentation);
                imageRepresentation.addDataInstance(vect);
                for (int i = 0; i < Math.min(descriptorLength,
                        lineItems.length); i++) {
                    vect.setDescriptorElement(i, Float.parseFloat(
                            lineItems[i]));
                }
                s = br.readLine();
            }
        }
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(keypointFile)))) {
            String s = br.readLine();
            String[] lineItems;
            int counter = 0;
            while (s != null) {
                lineItems = s.split(",");
                if (counter >= imageRepresentation.size()) {
                    break;
                }
                LFeatVector vect = (LFeatVector) imageRepresentation.
                        getInstance(counter);
                if (lineItems.length > 0) {
                    vect.setX(Float.parseFloat(lineItems[0]));
                }
                if (lineItems.length > 1) {
                    vect.setY(Float.parseFloat(lineItems[1]));
                }
                if (lineItems.length > 2) {
                    vect.setScale(Float.parseFloat(lineItems[2]));
                }
                if (lineItems.length > 3) {
                    vect.setAngle(Float.parseFloat(lineItems[3]));
                }
                s = br.readLine();
                counter++;
            }
        }
        return imageRepresentation;
    }
    
    /**
     * Saves the image feature representation into the keypoint and descriptor
     * file in the OpenCV format.
     * 
     * @param imageRepresentation LFeatRepresentation to save.
     * @param keypointFile File that will hold the keypoint specification for
     * the image.
     * @param descriptorFile File that will hold the descriptor value for the
     * image.
     * @throws IOException 
     */
    public static void writeRepresentationToFiles(
            LFeatRepresentation imageRepresentation, File keypointFile,
            File descriptorFile) throws IOException {
        if (keypointFile == null) {
            throw new NullPointerException("Null keypoint file target.");
        }
        if (descriptorFile == null) {
            throw new NullPointerException("Null descriptor file target.");
        }
        if (imageRepresentation == null) {
            return;
        }
        FileUtil.createFile(keypointFile);
        FileUtil.createFile(descriptorFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(descriptorFile))) {
            for (int i = 0; i < imageRepresentation.size(); i++) {
                LFeatVector vect = (LFeatVector) imageRepresentation.
                        getInstance(i);
                pw.print(vect.fAttr[4]);
                for (int j = 5; j < vect.fAttr.length; j++) {
                    pw.print("," + vect.fAttr[j]);
                }
                pw.println();
            }
        }
        try (PrintWriter pw = new PrintWriter(new FileWriter(keypointFile))) {
            for (int i = 0; i < imageRepresentation.size(); i++) {
                LFeatVector vect = (LFeatVector) imageRepresentation.
                        getInstance(i);
                pw.print(vect.getX());
                pw.print(",");
                pw.print(vect.getY());
                pw.print(",");
                pw.print(vect.getScale());
                pw.print(",");
                pw.print(vect.getAngle());
                pw.println();
            }
        }
    }
    
}
