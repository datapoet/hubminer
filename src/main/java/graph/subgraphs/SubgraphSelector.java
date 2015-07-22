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
import graph.basic.VertexInstance;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * This class implements the methods for subgraph selection based on the
 * provided vertex indexes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SubgraphSelector {

    /**
     * Generates the desired subgraph.
     *
     * @param g DMGraph g that is the original graph.
     * @param indexes int[] of selected indexes to generate the subgraph from.
     * @return DMGraph that is the desired subgraph.
     * @throws Exception
     */
    public static DMGraph getSubgraph(DMGraph g, int[] indexes)
            throws Exception {
        DMGraph subgraph = new DMGraph();
        subgraph.networkName = "A subgraph of " + g.networkName;
        subgraph.networkDescription = "Description of original graph: "
                + g.networkDescription;
        subgraph.vertices = new DataSet(g.vertices.iAttrNames,
                g.vertices.fAttrNames, g.vertices.sAttrNames);
        subgraph.vertices.data = new ArrayList(indexes.length);
        subgraph.edges = new DMGraphEdge[indexes.length];
        HashMap<Integer, Integer> subverticesHash = new HashMap<>(
                indexes.length * 3);
        // First handle the vertices.
        for (int selectedVertexIndex = 0; selectedVertexIndex < indexes.length;
                selectedVertexIndex++) {
            subgraph.vertices.data.add((VertexInstance) (g.vertices.data.get(
                    indexes[selectedVertexIndex])));
            // Populate the index map.
            subverticesHash.put(indexes[selectedVertexIndex],
                    selectedVertexIndex);
        }
        // Insert the edges incident to the selected indexes.
        for (int selectedVertexIndex = 0; selectedVertexIndex < indexes.length;
                selectedVertexIndex++) {
            DMGraphEdge edge = g.edges[indexes[selectedVertexIndex]];
            while (edge != null) {
                if (subverticesHash.containsKey(new Integer(edge.second))) {
                    DMGraphEdge subgraphEdge = new DMGraphEdge();
                    subgraphEdge.weight = edge.weight;
                    subgraphEdge.first = selectedVertexIndex;
                    subgraphEdge.second = subverticesHash.get(edge.second);
                    DMGraph.insertEdge(subgraph.edges, subgraphEdge);
                }
                edge = edge.next;
            }
        }
        return subgraph;
    }
}