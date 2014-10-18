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

import graph.basic.DMGraph;
import graph.basic.DMGraphEdge;
import java.util.ArrayList;

/**
 * This class calculates the connected components of a graph.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ConnectedComponents {

    private DMGraph g;
    private int numComponents;
    // Association of each vertex with a specific component.
    private int[] componentAssociationsArray;
    // Association list for each component.
    private ArrayList<ArrayList<Integer>> componentAssociationsVector;
    // An array of components as subgraphs.
    private DMGraph[] components;
    private boolean[] visitedVertices;
    private int currComponentCount = -1;

    /**
     * Initialization.
     *
     * @param g DMGraph that is to be analyzed.
     */
    public ConnectedComponents(DMGraph g) {
        this.g = g;
    }

    /**
     * @return Integer that is the number of connected components in the graph.
     */
    public int getNumComponents() {
        return numComponents;
    }

    /**
     * @return int[] that maps which vertices belong to which components.
     */
    public int[] getComponentAssociationsArray() {
        return componentAssociationsArray;
    }

    /**
     * @return ArrayList<ArrayList<Integer>> that is the list of lists of
     * indexes of vertices that belong to certain components.
     */
    public ArrayList<ArrayList<Integer>> getComponentAssociationsVector() {
        return componentAssociationsVector;
    }

    /**
     * @return DMGraph[] that are the graph's connected components.
     */
    public DMGraph[] getComponents() {
        return components;
    }

    public void findComponents() throws Exception {
        if ((g == null) || g.isEmpty()) {
            return;
        }
        int numVertices = g.vertices.data.size();
        // Initialize all the structures.
        visitedVertices = new boolean[numVertices];
        componentAssociationsArray = new int[numVertices];
        componentAssociationsVector = new ArrayList(10);
        // Go through all the nodes.
        for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
            if (!visitedVertices[vertexIndex]) {
                componentAssociationsVector.add(new ArrayList<Integer>(50));
                componentAssociationsArray[vertexIndex] = ++currComponentCount;
                visitedVertices[vertexIndex] = true;
                componentAssociationsVector.get(currComponentCount).add(
                        vertexIndex);
                // Here we traverse all that is reachable from the current node.
                visitComponent(currComponentCount, vertexIndex);
            }
        }
        numComponents = currComponentCount;
    }

    /**
     * Traverses the component from a node.
     *
     * @param componentIndex Integer that is the component index.
     * @param vertexIndex Integer that is the vertex index.
     * @throws Exception
     */
    private void visitComponent(int componentIndex, int vertexIndex)
            throws Exception {
        DMGraphEdge edge = g.edges[vertexIndex];
        while (edge != null) {
            if (!visitedVertices[edge.second]) {
                visitedVertices[edge.second] = true;
                componentAssociationsVector.get(componentIndex).add(
                        edge.second);
                componentAssociationsArray[edge.second] = componentIndex;
                visitComponent(componentIndex, edge.second);
            }
            edge = edge.next;
        }
    }

    /**
     * Generates all the component subgraphs.
     *
     * @throws Exception
     */
    public void generateGraphsForComponents() throws Exception {
        components = new DMGraph[numComponents];
        for (int componentIndex = 0; componentIndex < numComponents;
                componentIndex++) {
            int[] indexes = new int[componentAssociationsVector.get(
                    componentIndex).size()];
            for (int j = 0; j < indexes.length; j++) {
                indexes[j] = componentAssociationsVector.get(
                        componentIndex).get(j);
            }
            components[componentIndex] = SubgraphSelector.getSubgraph(g,
                    indexes);
            components[componentIndex].networkName = "Component "
                    + componentIndex;
            components[componentIndex].networkDescription =
                    componentIndex + "th component of graph " + g.networkName;
        }
    }
}