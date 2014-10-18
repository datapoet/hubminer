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
package images.mining.codebook;

import data.representation.images.quantized.QuantizedImageDistribution;
import data.representation.images.quantized.QuantizedImageDistributionDataSet;
import data.representation.images.quantized.QuantizedImageHistogram;
import data.representation.images.quantized.QuantizedImageHistogramDataSet;
import data.representation.images.sift.SIFTRepresentation;
import data.representation.images.sift.SIFTVector;
import distances.primary.SIFTMetric;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * SIFT features codebook class for feature quantization.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTCodeBook {

    public static final int DEFAULT_SIZE = 400;
    // Feature vectors that define the codeboook.
    private ArrayList<SIFTVector> codebook = new ArrayList<>();

    /**
     * @return Integer that is the codebook size.
     */
    public int getSize() {
        return codebook.size();
    }

    /**
     * Adds a new vector to the existing codebook.
     *
     * @param v SIFT feature vector to add to the codebook.
     */
    public void addVectorToCodeBook(SIFTVector v) {
        codebook.add(v);
    }

    /**
     * Sets the codebook.
     *
     * @param codebook ArrayList of SIFT feature vectors comprising the current
     * codebook.
     */
    public void setCodeBookSet(ArrayList<SIFTVector> codebook) {
        this.codebook = codebook;
    }

    /**
     * Generates a new QuantizedImageHistogramDataSet context corresponding to
     * this codebook representation.
     *
     * @return A new QuantizedImageHistogramDataSet context corresponding to
     * this codebook representation.
     */
    public QuantizedImageHistogramDataSet getNewHistogramContext() {
        if (codebook != null) {
            return new QuantizedImageHistogramDataSet(codebook.size());
        } else {
            return new QuantizedImageHistogramDataSet(DEFAULT_SIZE);
        }
    }

    /**
     * Generates a new QuantizedImageDistributionDataSet context corresponding
     * to this codebook representation.
     *
     * @return A new QuantizedImageDistributionDataSet context corresponding to
     * this codebook representation.
     */
    public QuantizedImageDistributionDataSet getNewDistributionContext() {
        if (codebook != null) {
            return new QuantizedImageDistributionDataSet(codebook.size());
        } else {
            return new QuantizedImageDistributionDataSet(DEFAULT_SIZE);
        }
    }

    /**
     * Generates a quantized image representation.
     *
     * @param rep SIFTRepresentation of the image to quantize.
     * @param qihDSet QuantizedImageHistogramDataSet data context to use for
     * initializing the quantized representation.
     * @return QuantizedImageHistogram that is the quantized image
     * representation.
     * @throws Exception
     */
    public QuantizedImageHistogram getHistogramForImageRepresentation(
            SIFTRepresentation rep,
            QuantizedImageHistogramDataSet qihDSet) throws Exception {
        QuantizedImageHistogram qih = new QuantizedImageHistogram(qihDSet);
        if (rep == null || rep.isEmpty()) {
            return qih;
        }
        qih.setPath(rep.getPath());
        for (int i = 0; i < rep.data.size(); i++) {
            qih.iAttr[getIndexOfClosestCodebook((SIFTVector) (
                    rep.data.get(i)))]++;
        }
        return qih;
    }

    /**
     * Generates a quantized image representation.
     *
     * @param rep SIFTRepresentation of the image to quantize.
     * @return QuantizedImageHistogram that is the quantized image
     * representation.
     * @throws Exception
     */
    public QuantizedImageHistogram getHistogramForImageRepresentation(
            SIFTRepresentation rep) throws Exception {
        QuantizedImageHistogramDataSet qihDSet =
                new QuantizedImageHistogramDataSet(codebook.size());
        QuantizedImageHistogram qih = new QuantizedImageHistogram(qihDSet);
        if (rep == null || rep.isEmpty()) {
            return null;
        }
        qih.setPath(rep.getPath());
        for (int i = 0; i < rep.data.size(); i++) {
            qih.iAttr[getIndexOfClosestCodebook((SIFTVector) (
                    rep.data.get(i)))]++;
        }
        return qih;
    }

    /**
     * Generates a quantized image representation, normalized to a probability
     * distribution.
     *
     * @param rep SIFTRepresentation of the image to quantize.
     * @return QuantizedImageHistogram that is the quantized image
     * representation, normalized to a probability distribution.
     * @throws Exception
     */
    public QuantizedImageDistribution getDistributionForImageRepresentation(
            SIFTRepresentation rep) throws Exception {
        QuantizedImageDistributionDataSet qidDSet =
                new QuantizedImageDistributionDataSet(codebook.size());
        QuantizedImageDistribution qid = new QuantizedImageDistribution(
                qidDSet);
        if (rep == null || rep.isEmpty()) {
            return null;
        }
        qid.setPath(rep.getPath());
        for (int i = 0; i < rep.data.size(); i++) {
            qid.fAttr[getIndexOfClosestCodebook((SIFTVector) (
                    rep.data.get(i)))]++;
        }
        // Normalization to get a probability distribution over codebooks.
        for (int i = 0; i < codebook.size(); i++) {
            qid.fAttr[i] /= rep.data.size();
        }
        return qid;
    }

    /**
     * Generates a quantized image representation, normalized to a probability
     * distribution.
     *
     * @param rep SIFTRepresentation of the image to quantize.
     * @param qidDSet QuantizedImageDistributionDataSet data context.
     * @return QuantizedImageHistogram that is the quantized image
     * representation, normalized to a probability distribution.
     * @throws Exception
     */
    public QuantizedImageDistribution getDistributionForImageRepresentation(
            SIFTRepresentation rep, QuantizedImageDistributionDataSet qidDSet)
            throws Exception {
        QuantizedImageDistribution qid =
                new QuantizedImageDistribution(qidDSet);
        if (rep != null) {
            qid.setPath(rep.getPath());
        }
        if (rep == null || rep.isEmpty()) {
            return qid;
        }
        for (int i = 0; i < rep.data.size(); i++) {
            qid.fAttr[getIndexOfClosestCodebook((SIFTVector) (
                    rep.data.get(i)))]++;
        }
        //normalization to get a probability distribution over codebooks
        for (int i = 0; i < codebook.size(); i++) {
            qid.fAttr[i] /= rep.data.size();
        }
        return qid;

    }

    /**
     * Returns the index of the closest codebook vector.
     *
     * @param vect SIFTVector to find the corresponding codebook vector for.
     * @return Integer that is the index of the closest codebook vector.
     * @throws Exception
     */
    public int getIndexOfClosestCodebook(SIFTVector vect) throws Exception {
        // Only the desciptors are taken into account in distance calculations.
        int closest = -1;
        SIFTMetric smet = new SIFTMetric();
        float currMinDist = Float.MAX_VALUE;
        float tempDist;
        for (int i = 0; i < codebook.size(); i++) {
            tempDist = smet.dist(vect, codebook.get(i));
            if (tempDist < currMinDist) {
                currMinDist = tempDist;
                closest = i;
            }
        }
        return closest;
    }

    /**
     * Persists the codebook data.
     *
     * @param outCodebookFile File to write the codebook to.
     * @throws Exception
     */
    public void writeCodeBookToFile(File outCodebookFile) throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(outCodebookFile));
        try {
            pw.println("codebook_size:" + codebook.size());
            for (SIFTVector sift : codebook) {
                pw.println(sift.toCSVString());
            }
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * Loads the codebook data from a file.
     *
     * @param inCodebookFile File to load the codebook data from.
     * @throws Exception
     */
    public void loadCodeBookFromFile(File inCodebookFile) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inCodebookFile)));
        try {
            String s = br.readLine();
            while ((s != null) && !(s.contains("codebook_size"))) {
                s = br.readLine();
            }
            if (s == null) {
                throw new Exception("Could not find 'codebook_size' within"
                        + " file: " + inCodebookFile.getPath());
            }
            s = s.trim();
            String[] pair = s.split(":");
            int size = Integer.parseInt(pair[1]);
            codebook = new ArrayList<>(size);
            s = br.readLine();
            SIFTVector tempVector;
            while (s != null) {
                tempVector = new SIFTVector();
                tempVector.fillFromCSVString(s);
                tempVector.setContext(null);
                codebook.add(tempVector);
                s = br.readLine();
            }
        } catch (Exception e) {
            throw e;
        } finally {
            br.close();
        }
    }
}
