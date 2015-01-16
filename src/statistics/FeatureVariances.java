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
package statistics;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import learning.unsupervised.Cluster;

/**
 * A class for calculating and representing feature variances.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class FeatureVariances {

    private DataSet dset;
    public float[] intVariances;
    public float[] floatVariances;

    /**
     * @param dataset DataSet object used in calculations.
     */
    public FeatureVariances(DataSet dataset) {
        this.dset = dataset;
        if (dataset == null || dataset.isEmpty()) {
            return;
        }
        intVariances = new float[dataset.getNumIntAttr()];
        floatVariances = new float[dataset.getNumFloatAttr()];
    }

    /**
     * @param c The corresponding Cluster.
     */
    public FeatureVariances(Cluster clust) {
        if (clust == null || clust.isEmpty()) {
            return;
        }
        this.dset = clust.getDefinitionDataset();
        intVariances = new float[this.dset.getNumIntAttr()];
        floatVariances = new float[this.dset.getNumFloatAttr()];
    }

    /**
     * @param clust Cluster that is to be analyzed.
     * @throws Exception
     */
    public void calculateAllVariances(Cluster clust) throws Exception {
        if (clust == null || clust.isEmpty()) {
            return;
        }
        DataInstance centroid = clust.getCentroid();
        int[] intCounts = new int[dset.getNumIntAttr()];
        int[] floatCounts = new int[dset.getNumFloatAttr()];
        float[] intSums = new float[dset.getNumIntAttr()];
        float[] floatSums = new float[dset.getNumFloatAttr()];
        for (int index = 0; index < dset.size(); index++) {
            DataInstance instance = dset.getInstance(index);
            for (int dint = 0; dint < dset.getNumIntAttr(); dint++) {
                if (DataMineConstants.isAcceptableInt(centroid.iAttr[dint])
                        && DataMineConstants.isAcceptableInt(
                        instance.iAttr[dint])) {
                    intCounts[dint]++;
                    intSums[dint] += Math.pow((instance.iAttr[dint] -
                            centroid.iAttr[dint]), 2);
                }
            }
            for (int dfloat = 0; dfloat < dset.getNumIntAttr(); dfloat++) {
                if (DataMineConstants.isAcceptableFloat(centroid.fAttr[dfloat])
                        && DataMineConstants.isAcceptableFloat(
                        instance.fAttr[dfloat])) {
                    floatCounts[dfloat]++;
                    floatSums[dfloat] += Math.pow((instance.fAttr[dfloat] -
                            centroid.fAttr[dfloat]), 2);
                }
            }
        }
        for (int dint = 0; dint < dset.getNumIntAttr(); dint++) {
            if (intCounts[dint] > 0) {
                intVariances[dint] = intSums[dint] / intCounts[dint];
            }
        }
        for (int dfloat = 0; dfloat < dset.getNumIntAttr(); dfloat++) {
            if (floatCounts[dfloat] > 0) {
                floatVariances[dfloat] = floatSums[dfloat] /
                        floatCounts[dfloat];
            }
        }
    }

    /**
     * Performs variance calculations.
     *
     * @throws Exception
     */
    public void calculateAllVariances() throws Exception {
        if (dset == null || dset.isEmpty()) {
            return;
        }
        calculateAllVariances(Cluster.fromEntireDataset(dset));
    }

    /**
     * @return Norm of the variance vector.
     */
    public float varianceNorm() {
        float sum = 0f;
        for (int dint = 0; dint < dset.getNumIntAttr(); dint++) {
            if (DataMineConstants.isAcceptableFloat(intVariances[dint])) {
                sum += Math.pow(Math.abs(intVariances[dint]), 2);
            }
        }
        for (int dfloat = 0; dfloat < dset.getNumFloatAttr(); dfloat++) {
            if (DataMineConstants.isAcceptableFloat(floatVariances[dfloat])) {
                sum += Math.pow(Math.abs(floatVariances[dfloat]), 2);
            }
        }
        return (float) Math.pow(sum, 0.5);
    }
}