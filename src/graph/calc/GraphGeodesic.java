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

import data.representation.util.DataMineConstants;
import graph.basic.DMGraphEdge;

/**
 * This class implements the methods for calculating the geodesic graph distance
 * between two vertices.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GraphGeodesic {

    /**
     * Default constructor.
     */
    public GraphGeodesic() {
    }

    /**
     * Calculates all the shortest distances between pairs of vertices via the
     * Floyd-Warshall algorithm.
     *
     * @param incidenceMatrix DMGraphEdge[] that is the incidence matrix.
     * @return double[][] that is the geodesic distance matrix.
     * @throws Exception
     */
    public double[][] calculateAllShortestDistances(
            DMGraphEdge[] incidenceMatrix) throws Exception {
        double[][] geodesicDistances = new double[incidenceMatrix.length][];
        for (int i = 0; i < incidenceMatrix.length; i++) {
            geodesicDistances[i] = new double[incidenceMatrix.length - i];
        }
        for (int i = 0; i < incidenceMatrix.length; i++) {
            for (int j = 1; j < geodesicDistances[i].length; j++) {
                geodesicDistances[i][i + j] = Double.MAX_VALUE;
            }
        }
        int numNodes = incidenceMatrix.length;
        for (int k = 0; k < numNodes; k++) {
            for (int i = 0; i < numNodes; i++) {
                for (int j = 0; j < numNodes; j++) {
                    double distIK = geodesicDistances[Math.min(i, k)][
                            Math.max(i, k) - Math.min(i, k)];
                    double distKJ = geodesicDistances[Math.min(j, k)][
                            Math.max(j, k) - Math.min(j, k)];
                    double distIJ = geodesicDistances[Math.min(i, j)][
                            Math.max(i, j) - Math.min(i, j)];
                    if (DataMineConstants.isAcceptableDouble(distIK + distKJ)
                            && distIK + distKJ < distIJ) {
                        geodesicDistances[Math.min(i, j)][Math.max(i, j)
                                - Math.min(i, j)] = distIK + distKJ;
                    }
                }
            }
        }
        return geodesicDistances;
    }
}