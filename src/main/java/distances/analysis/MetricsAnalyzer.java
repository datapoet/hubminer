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
package distances.analysis;

import java.util.Arrays;

import data.representation.DataSet;
import learning.supervised.Category;

/**
 * Analyzes the distance matrix.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MetricsAnalyzer {

    /**
     * @param distMat The distance matrix.
     * @return The percentage of triangles not satisfying the triangle
     * inequality.
     */
    public double getTriangleInequalityAssumptionBreach(float[][] distMat) {
        if (distMat == null || distMat.length == 0) {
            return 0;
        }
        int size = distMat.length;
        double incorrectSum = 0;
        int numTriangles = size * (size - 1) * (size - 2) / 6;
        // In each triangle, if there is a breach, there is only one breach - so
        // we are counting the number of correct triangles.
        float d1, d2, d3;
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                for (int k = j + 1; k < size; k++) {
                    d1 = distMat[i][j - i - 1];
                    d2 = distMat[j][k - j - 1];
                    d3 = distMat[i][k - i - 1];
                    if (d1 + d2 < d3) {
                        incorrectSum++;
                        break;
                    }
                    if (d1 + d3 < d2) {
                        incorrectSum++;
                        break;
                    }
                    if (d2 + d3 < d1) {
                        incorrectSum++;
                        break;
                    }
                }
            }
        }
        double result = incorrectSum / numTriangles;
        return result;
    }

    /**
     * It calculates an index which measures how concordant are the distances
     * with respect to data classes. Standardized categories are assumed. The
     * index takes a value betwen -1 and 1.
     *
     * @param distMat distance matrix
     * @param dset DataSet
     * @return GoodmanKruskalIndex = (Nc - Nd) / (Nc + Nd), Nc - concordant, Nd
     * - discordant distances
     */
    public float goodmanKruskalIndex(float[][] distMat,
            DataSet dset) throws Exception {
        if (distMat == null || distMat.length == 0) {
            return 0;
        }
        int[] labels = dset.obtainLabelArray();
        Category[] classes = dset.getClassesArray(dset.countCategories());
        int numIntraDists = 0;
        int numInterDists = 0;
        int numClasses = classes.length;
        if (numClasses < 2) {
            return 0;
        }
        for (int c1 = 0; c1 < numClasses; c1++) {
            if (classes[c1].size() > 0) {
                numIntraDists += classes[c1].size()
                        * (classes[c1].size() - 1) / 2;
                for (int c2 = 0; c2 < numClasses; c2++) {
                    if (classes[c2].size() > 0) {
                        numInterDists += classes[c1].size()
                                * classes[c2].size();
                    }
                }
            }
        }
        float[] intraDists = new float[numIntraDists];
        float[] interDists = new float[numInterDists];
        int intraIndex = -1;
        int interIndex = -1;
        for (int i = 0; i < distMat.length; i++) {
            for (int j = 0; j < distMat[i].length; j++) {
                if (labels[i] == labels[i + j + 1]) {
                    intraDists[++intraIndex] = distMat[i][j];
                } else {
                    interDists[++interIndex] = distMat[i][j];
                }
            }
        }
        Arrays.sort(intraDists); //ascending sort
        Arrays.sort(interDists); //ascending sort
        intraIndex = 0;
        interIndex = 0;
        int totalDists = numIntraDists + numInterDists;
        int totalPairs = numIntraDists * numInterDists;
        // This is the sum of concordant and discordant pairs.
        int Nc = 0; // Num concordant pairs.
        int Nd = 0; // Num discordant pairs.
        // We will only count the breaches.
        do {
            while (intraIndex < numIntraDists
                    && intraDists[intraIndex] < interDists[interIndex]) {
                intraIndex++;
            }
            if (intraIndex == numIntraDists) {
                break;
            }
            // Then the interDists[interIndex] is disconcordant with all
            // intraDists[i] for i >= intraIndex. There are numIntraDists -
            //intraIndex of them
            Nd += numIntraDists - intraIndex;
            // The inter-distance at interIndex has been processed, so it is
            // time to move on.
            interIndex++;
        } while (intraIndex < numIntraDists && interIndex < numInterDists);
        Nc = totalPairs - Nd;
        float gkIndex = (float) (Nc - Nd) / (float) (Nc + Nd);
        return gkIndex;
    }
}
