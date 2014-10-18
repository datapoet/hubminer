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
package learning.unsupervised.initialization;

import data.representation.DataInstance;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Random;

/**
 * Initialization method for K-means++.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PlusPlusSeeder {

    private int numClusters = 2;
    private ArrayList<DataInstance> data = null;
    // Metric object for distance calculations.
    private CombinedMetric cmet = null;
    private double[] cumulativeProbabilities = null;
    private float[] shortestDistances = null;
    private boolean[] isCentroidArray = null;
    private int currLen = 0;

    /**
     * @param numClusters Number of clusters.
     * @param data An array list of data instances.
     * @param cmet CombinedMetrics object.
     */
    public PlusPlusSeeder(int numClusters, ArrayList<DataInstance> data,
            CombinedMetric cmet) {
        this.numClusters = numClusters;
        this.data = data;
        this.cmet = cmet;
    }

    /**
     * Binary search within the cumulative probabilities array.
     *
     * @param searchValue Double representing the query value.
     * @param first Integer that is the current lower bound for the search.
     * @param second Integer that is the current upper bound for the search.
     * @return
     */
    public int findIndex(double searchValue, int first, int second) {
        if (second - first <= 1) {
            return second; //first isn't, so it must be second
        }
        int middle = (first + second) / 2;
        if (cumulativeProbabilities[middle] < searchValue) {
            return findIndex(searchValue, middle, second);
        } else {
            return findIndex(searchValue, first, middle);
        }
    }

    /**
     * A method that gets the indexes of instances that act as initial centroids
     *
     * @return An integer array of centroid indexes.
     * @throws Exception
     */
    public int[] getCentroidIndexes() throws Exception {
        if (numClusters < 0) {
            throw new Exception(
                    "Number of clusters must be positive, not " + numClusters);
        }
        if (data == null || data.isEmpty() || data.size() < numClusters) {
            return new int[0];
        }
        // A trivial case where each instance is in its own cluster.
        if (data.size() == numClusters) {
            int[] centroids = new int[numClusters];
            for (int i = 0; i < centroids.length; i++) {
                centroids[i] = i;
            }
        }
        // Now for the actual centroid initialization method.
        int[] centroids = new int[numClusters];
        cumulativeProbabilities = new double[data.size()];
        shortestDistances = new float[data.size()];
        for (int i = 0; i < data.size(); i++) {
            shortestDistances[i] = Float.MAX_VALUE;
        }
        isCentroidArray = new boolean[data.size()];
        Random randa = new Random();
        int first = randa.nextInt(data.size());
        isCentroidArray[first] = true;
        currLen++;
        centroids[0] = first;
        float tmpDist;
        float randomizer;
        double searchValue;
        int nextCentroid;
        for (int i = 1; i < numClusters; i++) {
            // Calculate all (new) shortest and cumulative distances.
            for (int j = 0; j < data.size(); j++) {
                if (isCentroidArray[j]) {
                    shortestDistances[j] = 0;
                    if (j >= 1) {
                        cumulativeProbabilities[j] =
                                cumulativeProbabilities[j - 1];
                    } else {
                        cumulativeProbabilities[j] = 0;
                    }
                } else {
                    for (int k = 0; k < currLen; k++) {
                        tmpDist = cmet.dist(data.get(j),
                                data.get(centroids[k]));
                        if (tmpDist < shortestDistances[j]) {
                            shortestDistances[j] = tmpDist;
                        }
                        if (j >= 1) {
                            cumulativeProbabilities[j] =
                                    cumulativeProbabilities[j - 1]
                                    + shortestDistances[j] *
                                    shortestDistances[j];
                        } else {
                            cumulativeProbabilities[j] = shortestDistances[j];
                        }
                    }
                }
            }
            // Now a random guess for the new centroid, based on the currently
            // selected centroid set.
            randomizer = randa.nextFloat();
            searchValue = randomizer * cumulativeProbabilities[data.size() - 1];
            if (cumulativeProbabilities[0] > searchValue) {
                nextCentroid = 0;
                centroids[currLen++] = nextCentroid;
            } else {
                nextCentroid = findIndex(searchValue, 0, data.size() - 1);
                isCentroidArray[nextCentroid] = true;
                centroids[currLen++] = nextCentroid;
            }
        }
        return centroids;
    }
}
