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
package graph.subgraphs;

import data.representation.DataSet;
import graph.basic.DMGraph;
import graph.basic.DMGraphEdge;

/**
 * This class implements the functionality for a graph cut.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GraphCut {

    DMGraph g;

    public GraphCut(DMGraph g) {
        this.g = g;
    }

    /**
     * Generates a graph cut.
     *
     * @param minWeight Minimal admissible weight for an edge.
     * @param minDegree Minimal admissible degree for a node.
     * @return DMGraph that is the desired graph cut.
     * @throws Exception
     */
    public DMGraph generateGraphCut(double minWeight, int minDegree)
            throws Exception {
        DMGraphEdge[] retainedEdges = new DMGraphEdge[g.edges.length];
        DataSet retainedVertices = new DataSet(g.vertices.iAttrNames,
                g.vertices.fAttrNames, g.vertices.sAttrNames,
                g.vertices.data.size());
        DMGraph graphCut = new DMGraph("A cut (" + minWeight + ","
                + minDegree + ") of " + g.networkName, g.networkDescription,
                retainedVertices, retainedEdges);
        // First the edge weight filtering.
        for (int i = 0; i < g.edges.length; i++) {
            DMGraphEdge edge = g.edges[i];
            while (edge != null) {
                if (edge.weight >= minWeight) {
                    DMGraph.insertEdge(graphCut.edges, edge.copy());
                }
                edge = edge.next;
            }
        }
        // Now we will do some node deletions. We will have to keep track of
        // the new node index values in order to apply them to the edges
        // afterwards.
        int[] indexAliases = new int[g.edges.length];
        for (int i = 0; i < indexAliases.length; i++) {
            indexAliases[i] = i;
        }
        int countRemovedNodes = 0;
        for (int i = 0; i < g.size(); i++) {
            if (graphCut.getNodeDegree(i) < minDegree) {
                indexAliases[i] = -1;
                countRemovedNodes++;
            } else {
                indexAliases[i] -= countRemovedNodes;
            }
        }
        DMGraphEdge[] retainEdgesUpdatedIndexes =
                new DMGraphEdge[g.edges.length - countRemovedNodes];
        graphCut.edges = retainEdgesUpdatedIndexes;
        for (int i = 0; i < g.edges.length; i++) {
            if (indexAliases[i] > -1) {
                retainedVertices.addDataInstance(g.vertices.data.get(i));
                DMGraphEdge edge = retainedEdges[i].copy();
                while (edge != null) {
                    edge.first = indexAliases[edge.first];
                    edge.second = indexAliases[edge.second];
                    if ((edge.first > -1) && (edge.second > -1)) {
                        DMGraph.insertEdge(retainEdgesUpdatedIndexes,
                                retainedEdges[i]);
                    }
                    edge = edge.next;
                }
            }
        }
        return graphCut;
    }
}