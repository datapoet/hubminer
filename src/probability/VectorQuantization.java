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
package probability;

import data.representation.DataInstance;
import data.representation.DataSet;
import java.util.Random;

/**
 * A vector quantization approximation of the underlying probability
 * distribution. vector quantization algorithm
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class VectorQuantization {

    // Number of points to use in quantization.
    int numPoints = 100;
    int numSamplings;
    float[][] bounds;
    float alpha; // Temperature factor that governs convergence.
    DataSet dset;

    /**
     * @param dset DataSet object.
     * @param numPoints Number of points to use in quantization.
     * @param numSamplings Duration in terms of sampling.
     * @param bounds Bounds for the quantized vectors.
     */
    public VectorQuantization(
            DataSet dset,
            int numPoints,
            int numSamplings,
            float[][] bounds) {
        this.dset = dset;
        this.numPoints = numPoints;
        this.numSamplings = numSamplings;
        this.bounds = bounds;
    }

    /**
     * Perform vector quantization.
     *
     * @return Codebook produced by VQ.
     */
    public float[][] getCodebook() {
        DataInstance instance;
        instance = dset.data.get(0);
        float[][] codebook = new float[numPoints][instance.fAttr.length];
        alpha = 0.8f;
        float aFinal = 0.05f;
        float fact = (float) Math.pow((aFinal / alpha),
                1f / (float) numSamplings);
        // Random initialization.
        Random randa = new Random();
        for (int i = 0; i < codebook.length; i++) {
            for (int j = 0; j < codebook[i].length; j++) {
                codebook[i][j] =
                        bounds[j][0]
                        + randa.nextFloat() * (bounds[j][1] - bounds[j][0]);
            }
        }
        int index;
        float currDist;
        float minDist;
        int minIndex = 0;
        for (int i = 0; i < numSamplings; i++) {
            index = randa.nextInt(dset.size());
            instance = dset.data.get(index);
            // Find the closest codebook.
            minDist = Float.MAX_VALUE;
            for (int j = 0; j < codebook.length; j++) {
                currDist = 0;
                for (int k = 0; k < instance.fAttr.length; k++) {
                    currDist += (instance.fAttr[k] - codebook[j][k])
                            * (instance.fAttr[k] - codebook[j][k]);
                }
                if (currDist < minDist) {
                    minDist = currDist;
                    minIndex = j;
                }
            }
            // Move the closest codebook toward the data point.
            for (int j = 0; j < codebook[minIndex].length; j++) {
                codebook[minIndex][j] +=
                        alpha * (instance.fAttr[j] - codebook[minIndex][j]);
            }
            alpha *= fact;
        }
        return codebook;
    }

    /**
     * Perform vector quantization.
     *
     * @return Codebook produced by VQ.
     */
    public DataSet getCodebookInstances() {
        float[][] fArr = getCodebook();
        DataSet resCol = new DataSet();
        resCol.fAttrNames = new String[fArr[0].length];
        for (int i = 0; i < resCol.fAttrNames.length; i++) {
            resCol.fAttrNames[i] = "floatAtt" + i;
        }
        DataInstance instance;
        for (int i = 0; i < fArr.length; i++) {
            instance = new DataInstance(resCol);
            instance.embedInDataset(resCol);
            resCol.addDataInstance(instance);
            for (int j = 0; j < instance.fAttr.length; j++) {
                instance.fAttr[j] = fArr[i][j];
            }
        }
        return resCol;
    }
}
