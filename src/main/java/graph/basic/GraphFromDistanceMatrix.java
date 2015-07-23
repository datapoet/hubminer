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
package graph.basic;

import data.representation.DataSet;
import java.util.Arrays;

/**
 * This class implements the construction of a graph from the distance matrix on
 * the data by means of establishing an edge between points that are close. The
 * cut-off distance threshold can either be provided directly or set
 * dynamically. The edge weight can be either set to the distance weight or its
 * complement - the similarity. For this to work automatically, the distances
 * need to be normalized to the [0,1] range.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GraphFromDistanceMatrix {

    // Whether the edge weights will contain the similarity between vertices or
    // the distance values.
    public static final boolean SIMILARITY_GRAPH = true;
    // The default number of distances to accept for edge insertions.
    public static final float CUTOFF_PERCENTILE = 0.05f;
    // The actual proportion of distances to accept.
    private double cutoffPercentile = CUTOFF_PERCENTILE;
    // The distance threshold for the cut-off.
    private double threshold = 0.5;
    // Whether to set the distance threshold dynamically to correspond to the
    // cutoff percentile or not.
    private boolean setDynamically = true;
    // Graph vertices and the distance matrix.
    private DataSet vertices;
    public float[][] distMatrix;

    /**
     * Initialization.
     *
     * @param vertices DataSet object holding the vertices.
     * @param distMatrix float[][] holding the distance matrix.
     * @param setDynamically Boolean flag indicating whether to set the cut-off
     * threshold dynamically.
     */
    public GraphFromDistanceMatrix(DataSet vertices, float[][] distMatrix,
            boolean setDynamically) {
        this.setDynamically = setDynamically;
        this.vertices = vertices;
        this.distMatrix = distMatrix;
    }

    /**
     * Initialization.
     *
     * @param vertices DataSet object holding the vertices.
     * @param distMatrix float[][] holding the distance matrix.
     * @param setDynamically Boolean flag indicating whether to set the cut-off
     * threshold dynamically.
     * @param cutoffPercentile Double value that is the proportion of edges to
     * build.
     */
    public GraphFromDistanceMatrix(DataSet vertices, float[][] distMatrix,
            boolean setDynamically, double cutoffPercentile) {
        this.setDynamically = setDynamically;
        this.vertices = vertices;
        this.distMatrix = distMatrix;
        this.cutoffPercentile = cutoffPercentile;
    }

    /**
     * Initialization.
     *
     * @param vertices DataSet object holding the vertices.
     * @param distMatrix float[][] holding the distance matrix.
     * @param threshold Double that is the cut-off threshold for the distances.
     */
    public GraphFromDistanceMatrix(DataSet vertices, float[][] distMatrix,
            double threshold) {
        this.threshold = threshold;
        setDynamically = false;
        this.vertices = vertices;
        this.distMatrix = distMatrix;
    }

    /**
     * Generates an array that holds the distance matrix data.
     *
     * @return float[] holding the distances from the original distance matrix.
     */
    private float[] matrixToArray() {
        if (distMatrix == null) {
            return null;
        }
        float[] distArray = new float[distMatrix.length
                * (distMatrix.length - 1) / 2];
        int index = -1;
        for (int i = 0; i < distMatrix.length; i++) {
            for (int j = 0; j < distMatrix[i].length; j++) {
                distArray[++index] = distMatrix[i][j];
            }
        }
        return distArray;
    }

    /**
     * Gets the distance cutoff threshold for the specified percentile.
     *
     * @param perc Float value that is the desired percentile.
     * @return Float that is the cut-off threshold for the distances.
     */
    public float getThresholdForPercentile(float perc) {
        float[] array = matrixToArray();
        if (array == null) {
            return 0;
        }
        // Ascending sort.
        Arrays.sort(array);
        int keyIndex = (int) (perc * (float) (array.length));
        return array[keyIndex];
    }

    /**
     * Loads the graph from the provided distance matrix.
     *
     * @return DMGraph generate from the distance matrix via distance cut-off
     * thresholds.
     * @throws Exception
     */
    public DMGraph loadFromDistanceMatrix() throws Exception {
        if (vertices == null || vertices.isEmpty()) {
            return null;
        }
        DMGraph resultingGraph = new DMGraph();
        if (setDynamically) {
            threshold = getThresholdForPercentile((float) cutoffPercentile);
        }
        DMGraphEdge[] edges = new DMGraphEdge[vertices.size()];
        DMGraphEdge edge;
        for (int i = 0; i < distMatrix.length; i++) {
            for (int j = 0; j < distMatrix[i].length; j++) {
                // Check whether to insert the edge for the current pair.
                if (distMatrix[i][j] < threshold) {
                    edge = new DMGraphEdge();
                    if (SIMILARITY_GRAPH) {
                        edge.weight = 1f - distMatrix[i][j];
                    } else {
                        edge.weight = distMatrix[i][j];
                    }
                    edge.first = i;
                    edge.second = i + j + 1;
                    DMGraph.insertEdge(edges, edge);
                }
            }
        }
        resultingGraph.vertices = vertices;
        resultingGraph.edges = edges;
        return resultingGraph;
    }
}