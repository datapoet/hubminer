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

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.images.quantized.QuantizedImageDistribution;
import data.representation.images.quantized.QuantizedImageDistributionDataSet;
import data.representation.images.quantized.QuantizedImageHistogram;
import data.representation.images.quantized.QuantizedImageHistogramDataSet;
import distances.primary.CombinedMetric;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * A generic class for representing image feature codebooks.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GenericCodeBook {

    public static final int DEFAULT_SIZE = 400;
    // Feature vectors that define the codeboook.
    private ArrayList<DataInstance> codebook = new ArrayList<>();
    private CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;

    /**
     * @param cmet CombinedMetric object for distance calculations.
     */
    public void setCombinedMetirc(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @return CombinedMetric object for distance calculations.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    /**
     * @return Integer that is the codebook size.
     */
    public int getSize() {
        return codebook.size();
    }

    /**
     * Adds a new vector to the existing codebook.
     *
     * @param instance DataInstance to add to the codebook.
     */
    public void addVectorToCodeBook(DataInstance instance) {
        codebook.add(instance);
    }

    /**
     * Sets the codebook.
     *
     * @param codebook ArrayList of DataInstance objects comprising the current
     * codebook.
     */
    public void setCodeBookSet(ArrayList<DataInstance> codebook) {
        this.codebook = codebook;
    }

    /**
     * Generates a new QuantizedImageHistogramDataSet context corresponding to
     * this codebook representation.
     *
     * @return A new QuantizedImageHistogramDataSet context corresponding to
     * this codebook representation.
     */
    public DataSet getNewHistogramContext() {
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
     * @param rep DataSet that is the representation of the image to quantize.
     * @param qihDSet QuantizedImageHistogramDataSet data context to use for
     * initializing the quantized representation.
     * @return QuantizedImageHistogram that is the quantized image
     * representation.
     * @param imagePath String that is the path to the image file.
     * @throws Exception
     */
    public QuantizedImageHistogram getHistogramForImageRepresentation(
            DataSet rep, QuantizedImageHistogramDataSet qihDSet,
            String imagePath) throws Exception {
        QuantizedImageHistogram qih = new QuantizedImageHistogram(qihDSet);
        qih.setPath(imagePath);
        if (rep == null || rep.isEmpty()) {
            return qih;
        }
        for (int i = 0; i < rep.data.size(); i++) {
            qih.iAttr[getIndexOfClosestCodebook(rep.data.get(i))]++;
        }
        return qih;
    }

    /**
     * Generates a quantized image representation.
     *
     * @param rep DataSet that is the representation of the image to quantize.
     * @return QuantizedImageHistogram that is the quantized image
     * representation.
     * @param imagePath String that is the path to the image file.
     * @throws Exception
     */
    public QuantizedImageHistogram getHistogramForImageRepresentation(
            DataSet rep, String imagePath) throws Exception {
        QuantizedImageHistogramDataSet qihDSet =
                new QuantizedImageHistogramDataSet(codebook.size());
        QuantizedImageHistogram qih = new QuantizedImageHistogram(qihDSet);
        qih.setPath(imagePath);
        if (rep == null || rep.isEmpty()) {
            return qih;
        }
        for (int i = 0; i < rep.data.size(); i++) {
            qih.iAttr[getIndexOfClosestCodebook(rep.data.get(i))]++;
        }
        return qih;
    }

    /**
     * Generates a quantized image representation, normalized to a probability
     * distribution.
     *
     * @param rep DataSet that is the representation of the image to quantize.
     * @return QuantizedImageHistogram that is the quantized image
     * representation, normalized to a probability distribution.
     * @param imagePath String that is the path to the image file.
     * @throws Exception
     */
    public QuantizedImageDistribution getDistributionForImageRepresentation(
            DataSet rep, String imagePath) throws Exception {
        QuantizedImageDistributionDataSet qidDSet =
                new QuantizedImageDistributionDataSet(codebook.size());
        QuantizedImageDistribution qid = new QuantizedImageDistribution(
                qidDSet);
        qid.setPath(imagePath);
        if (rep == null || rep.isEmpty()) {
            return qid;
        }
        for (int i = 0; i < rep.data.size(); i++) {
            qid.fAttr[getIndexOfClosestCodebook(rep.data.get(i))]++;
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
     * @param rep DataSet that is the representation of the image to quantize.
     * @param qidDSet QuantizedImageDistributionDataSet data context.
     * @return QuantizedImageHistogram that is the quantized image
     * representation, normalized to a probability distribution.
     * @param imagePath String that is the path to the image file.
     * @throws Exception
     */
    public QuantizedImageDistribution getDistributionForImageRepresentation(
            DataSet rep, QuantizedImageDistributionDataSet qidDSet,
            String imagePath) throws Exception {
        QuantizedImageDistribution qid =
                new QuantizedImageDistribution(qidDSet);
        if (rep != null) {
            qid.setPath(imagePath);
        }
        if (rep == null || rep.isEmpty()) {
            return qid;
        }
        for (int i = 0; i < rep.data.size(); i++) {
            qid.fAttr[getIndexOfClosestCodebook(rep.data.get(i))]++;
        }
        // Normalization to get a probability distribution over codebooks.
        for (int i = 0; i < codebook.size(); i++) {
            qid.fAttr[i] /= rep.data.size();
        }
        return qid;
    }

    /**
     * Returns the index of the closest codebook vector.
     *
     * @param instance DataInstance to find the corresponding codebook vector
     * for.
     * @return Integer that is the index of the closest codebook vector.
     * @throws Exception
     */
    public int getIndexOfClosestCodebook(DataInstance instance)
            throws Exception {
        // Only the desciptors are taken into account in distance calculations.
        int closest = -1;
        float currMinDist = Float.MAX_VALUE;
        float tempDist;
        for (int i = 0; i < codebook.size(); i++) {
            tempDist = cmet.dist(instance, codebook.get(i));
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
            for (DataInstance instance : codebook) {
                pw.println(instance.floatsToCSVString());
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
            DataInstance instance;
            while (s != null) {
                instance = new DataInstance();
                String[] parts = s.split(",");
                instance.fAttr = new float[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    instance.fAttr[i] = Float.parseFloat(parts[i]);
                }
                instance.embedInDataset(null);
                codebook.add(instance);
                s = br.readLine();
            }
        } catch (Exception e) {
            throw e;
        } finally {
            br.close();
        }
    }
}
