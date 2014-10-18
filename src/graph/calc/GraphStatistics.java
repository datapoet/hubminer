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
package graph.calc;

import data.representation.util.DataSortableInfo;
import data.representation.util.DataSortableInfoComparator;
import graph.basic.DMGraph;
import graph.basic.VertexInstance;
import java.util.Arrays;

/**
 * This class implements the methods for calculating some basic network
 * properties like closeness, centrality, etc.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GraphStatistics {

    // Types of weights.
    public static final boolean SIMILARITY = false;
    public static final boolean DISSIMILARITY = true;

    /**
     * Calculates a list of nodes ordered by scale, which is returned as a
     * DataSortableInfo array.
     *
     * @param g DMGraph to analyze.
     * @return DataSortableInfo[] that represents the vertices ordered by scale.
     * @throws Exception
     */
    public static DataSortableInfo[] sortedByScale(DMGraph g) throws Exception {
        if (g.isEmpty()) {
            return new DataSortableInfo[0];
        }
        DataSortableInfo[] sortedInfo =
                new DataSortableInfo[g.vertices.data.size()];
        for (int i = 0; i < sortedInfo.length; i++) {
            sortedInfo[i].primitiveSortable = ((VertexInstance)
                    (g.vertices.data.get(i))).scale;
            sortedInfo[i].originalInstance = ((VertexInstance)
                    (g.vertices.data.get(i)));
            sortedInfo[i].originalIndex = i;
        }
        DataSortableInfoComparator comparator = new DataSortableInfoComparator(
                DataSortableInfoComparator.SIMPLE);
        Arrays.sort(sortedInfo, comparator);
        return sortedInfo;
    }

    /**
     * Calculates the degree centrality for all the vertices.
     *
     * @param g DMGraph that is to be analyzed.
     * @return double[] of degree centrality scores for all the nodes in the
     * graph.
     * @throws Exception
     */
    public static double[] degreeCentralityList(DMGraph g) throws Exception {
        if (g.isEmpty()) {
            return null;
        }
        double[] degreeCentrality = new double[g.edges.length];
        for (int i = 0; i < g.edges.length; i++) {
            degreeCentrality[i] = g.getNodeDegree(i) / (g.edges.length - 1);
        }
        return degreeCentrality;
    }

    /**
     * Calculates the degree centrality for all the vertices.
     *
     * @param g DMGraph that is to be analyzed.
     * @return DataSortableInfo[] that represents the vertices ordered by their
     * degree centrality.
     * @throws Exception
     */
    public static DataSortableInfo[] degreeCentrality(DMGraph g)
            throws Exception {
        double[] degreeCentrality = degreeCentralityList(g);
        if (degreeCentrality == null) {
            return null;
        }
        DataSortableInfo[] degreeCentralityInfo =
                new DataSortableInfo[degreeCentrality.length];
        for (int i = 0; i < degreeCentralityInfo.length; i++) {
            degreeCentralityInfo[i] = new DataSortableInfo(
                    i, g.vertices.data.get(i), degreeCentrality[i]);
        }
        DataSortableInfoComparator comparator = new DataSortableInfoComparator(
                DataSortableInfoComparator.SIMPLE);
        Arrays.sort(degreeCentralityInfo, comparator);
        return degreeCentralityInfo;
    }

    /**
     * Calculates the closeness centrality for all the vertices.
     *
     * @param g DMGraph that is to be analyzed.
     * @return DataSortableInfo[] that represents the vertices ordered by their
     * closeness centrality.
     * @throws Exception
     */
    public static DataSortableInfo[] closenessCentrality(
            DMGraph g) throws Exception {
        double[] closenessCentrality = closenessCentralityList(g);
        if (closenessCentrality == null) {
            return null;
        }
        DataSortableInfo[] closenessCentralityInfo =
                new DataSortableInfo[closenessCentrality.length];
        for (int i = 0; i < closenessCentralityInfo.length; i++) {
            closenessCentralityInfo[i] = new DataSortableInfo(
                    i, g.vertices.data.get(i), closenessCentrality[i]);
        }
        DataSortableInfoComparator comparator = new DataSortableInfoComparator(
                DataSortableInfoComparator.SIMPLE);
        Arrays.sort(closenessCentralityInfo, comparator);
        return closenessCentralityInfo;
    }

    /**
     * Calculates the closeness centrality for all the vertices. Closeness for a
     * vertex is an inverse of the average geodesic distance from that to all
     * the other vertices in the graph. However, the definition of closeness by
     * Galnachev allows for calculating closeness for disconnected graphs. It is
     * then sum_(t in V\v) 2^(-d(v,t)) where d(v,t) denotes the geodesic
     * distance between v and t.
     *
     * @param g DMGraph that is to be analyzed.
     * @return double[] of closeness centrality scores for all the nodes in the
     * graph.
     * @throws Exception
     */
    public static double[] closenessCentralityList(DMGraph g) throws Exception {
        GraphGeodesic geodesicCalculator = new GraphGeodesic();
        double[][] shortestDist =
                geodesicCalculator.calculateAllShortestDistances(g.edges);
        double[] closenessCntralityArray = new double[g.edges.length];
        for (int i = 0; i < g.edges.length; i++) {
            for (int j = i + 1; j < g.edges.length; j++) {
                if (shortestDist[i][j] != Double.MAX_VALUE) {
                    closenessCntralityArray[i] += (1 / Math.pow(2,
                            shortestDist[i][j]));
                    closenessCntralityArray[j] += (1 / Math.pow(2,
                            shortestDist[i][j]));
                }
            }
        }
        return closenessCntralityArray;
    }

    /**
     * This method calculates the network degree centralization measure.
     *
     * @param g DMGraph g that is to be analyzed.
     * @return Double that is the network degree centralization.
     * @throws Exception
     */
    public static double networkDegreeCentralization(DMGraph g)
            throws Exception {
        int maxDegree = g.getMaxNodeDegree();
        int deltaToMaxSum = g.edges.length * maxDegree;
        for (int i = 0; i < g.edges.length; i++) {
            deltaToMaxSum -= g.getNodeDegree(i);
        }
        double centralization = (double) deltaToMaxSum /
                (double) ((g.edges.length - 1) * (g.edges.length - 2));
        return centralization;
    }
}