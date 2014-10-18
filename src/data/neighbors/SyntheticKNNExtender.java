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
package data.neighbors;

import data.generators.BasicGaussianDatasetExtender;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;

/**
 * This class implements the methods for improving the estimates in the neighbor
 * occurrence models by introducing synthetic data points as queries and
 * querying the original dataset more times.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SyntheticKNNExtender {

    private ArrayList<DataInstance> syntheticExt;
    // Separate arrays for the kNN sets on synthetic data instances.
    private int[][] syntheticKNeighbors;
    private float[][] syntheticKDistances;
    private BasicGaussianDatasetExtender dataExtender = null;
    private int k;
    private DataSet dset;
    private CombinedMetric cmet;
    private int[] kNeighborFrequencies;
    private int[] kGoodFrequencies;
    private int[] kBadFrequencies;
    private float[][] synthClassConditionalOccCounts;
    private int numClasses;

    /**
     * Initialization.
     *
     * @param dset DataSet object to extend and improve the occurrence estimates
     * on.
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public SyntheticKNNExtender(DataSet dset, int k, CombinedMetric cmet,
            int numClasses) {
        this.dset = dset;
        this.k = k;
        this.cmet = cmet;
        this.numClasses = numClasses;
    }

    /**
     * @return int[] that is the additional occurrence frequencies of the
     * original points on the synthetic Gaussian extension of the data.
     */
    public int[] getSyntheticOccFreqs() {
        return kNeighborFrequencies;
    }

    /**
     * @return int[] that is the additional good occurrence frequencies of the
     * original points on the synthetic Gaussian extension of the data.
     */
    public int[] getSyntheticGoodOccFreqs() {
        return kGoodFrequencies;
    }

    /**
     * @return int[] that is the additional bad occurrence frequencies of the
     * original points on the synthetic Gaussian extension of the data.
     */
    public int[] getSyntheticBadOccFreqs() {
        return kBadFrequencies;
    }

    /**
     * @return float[][] that is the additional class-conditional neighbor
     * occurrence frequencies of the original points on the synthetic Gaussian
     * extension of the data.
     */
    public float[][] getSyntheticClassConditionalOccCounts() {
        return synthClassConditionalOccCounts;
    }

    /**
     * Extends the data for evaluating the occurrence frequencies of the
     * original data points.
     *
     * @param numSyntheticPoints Integer that is the number of additional points
     * to generate.
     */
    public void extendData(int numSyntheticPoints) {
        if (dataExtender == null) {
            dataExtender = new BasicGaussianDatasetExtender(dset);
            dataExtender.generateGaussianModel();
        }
        syntheticExt = dataExtender.generateSyntheticInstances(
                numSyntheticPoints);
    }

    /**
     * Calculate the kNN sets on the synthetic data extension.
     *
     * @throws Exception
     */
    public void calcSyntheticNSets() throws Exception {
        if (syntheticExt != null && syntheticExt.size() > 0) {
            syntheticKNeighbors = new int[syntheticExt.size()][];
            syntheticKDistances = new float[syntheticExt.size()][k];
            kNeighborFrequencies = new int[dset.size()];
            kGoodFrequencies = new int[dset.size()];
            kBadFrequencies = new int[dset.size()];
            synthClassConditionalOccCounts = new float[numClasses][dset.size()];
            for (int i = 0; i < syntheticExt.size(); i++) {
                syntheticKNeighbors[i] =
                        NeighborSetFinder.getIndexesOfNeighbors(dset,
                        syntheticExt.get(i), k, cmet);
                for (int kInd = 0; kInd < k; kInd++) {
                    syntheticKDistances[i][kInd] = cmet.dist(
                            syntheticExt.get(i), dset.data.get(
                            syntheticKNeighbors[i][kInd]));
                    kNeighborFrequencies[syntheticKNeighbors[i][kInd]]++;
                    synthClassConditionalOccCounts[
                            syntheticExt.get(i).getCategory()][
                            syntheticKNeighbors[i][kInd]]++;
                    if (dset.data.get(
                            syntheticKNeighbors[i][kInd]).getCategory()
                            == syntheticExt.get(i).getCategory()) {
                        kGoodFrequencies[syntheticKNeighbors[i][kInd]]++;
                    } else {
                        kBadFrequencies[syntheticKNeighbors[i][kInd]]++;
                    }
                }
            }
        }
    }
}
