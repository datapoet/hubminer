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
import org.jgraph.JGraph;

/**
 * This class offers one simple graph implementation that can be used in
 * analysis and visualization. The latter is also the reason why a JGraph object
 * is associated with a DMGraph object which essentially provides a data model
 * to the former in cases of visualization. Vertices are represented via a
 * DataSet object. This implementation is used mostly to analyze small networks
 * and visualize them and is not meant for large-scale data analysis. Graph
 * analysis is not the focus of this library.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DMGraph {

    public DataSet vertices;
    public DMGraphEdge[] edges;
    public String networkName;
    public String networkDescription;
    // Copying a DMGraph object will not copy the JGraph reference, by default.
    // JGraph is only used when the results are to be displayed on the screen.
    // JGraph is not always used for graph drawing in this library, so this is
    // more of a backward-compatibility thing.
    public JGraph visGraph = null;

    /**
     * The empty constructor.
     */
    public DMGraph() {
    }

    /**
     * Initialization.
     *
     * @param networkName String that is the name of the modeled network.
     * @param networkDescription String that is the description of the modeled
     * network.
     */
    public DMGraph(String networkName, String networkDescription) {
        this.networkName = networkName;
        this.networkDescription = networkDescription;
    }

    /**
     * Initialization.
     *
     * @param networkName String that is the name of the modeled network.
     * @param networkDescription String that is the description of the modeled
     * network.
     * @param vertices DataSet representing graph vertices. Each DataInstance is
     * a vertex in the graph.
     * @param edges DMGraphEdge array of graph edges.
     */
    public DMGraph(String networkName, String networkDescription,
            DataSet vertices, DMGraphEdge[] edges) {
        this.networkName = networkName;
        this.networkDescription = networkDescription;
        this.vertices = vertices;
        this.edges = edges;
    }

    /**
     * @return Integer that is the number of vertices in the graph.
     */
    public int size() {
        if (vertices == null) {
            return 0;
        } else {
            return vertices.size();
        }
    }

    /**
     * Rescale the vertex sizes.
     *
     * @param factor Double that is the factor to rescale the vertices by.
     */
    public void rescale(double factor) {
        for (int i = 0; i < vertices.size(); i++) {
            VertexInstance v = (VertexInstance) (vertices.data.get(i));
            v.scale *= factor;
        }
    }

    /**
     * Merges this graph with another one. However, this method changes both of
     * these graphs along the way! This is because it changes some of the
     * elements that are now contained in the merged graph, so this changes the
     * references in both of the original graphs as well. If this is
     * unacceptable, copies should be made and merged instead.
     *
     * @param g DMGraph to merge this one with.
     * @return DMGraph that is the result of merging two graphs.
     */
    public DMGraph merge(DMGraph g) {
        if (isEmpty()) {
            return g;
        }
        if ((g == null) || g.isEmpty()) {
            return this;
        }
        DMGraph mergedGraph = new DMGraph();
        mergedGraph.networkName = "A merge of " + networkName + " and "
                + g.networkName;
        mergedGraph.networkDescription = "Combined descriptions: "
                + networkDescription + " ||||| " + g.networkDescription;
        // Handle the headers in the vertice sets.
        String[] intAttrNames = null;
        String[] floatAttrNames = null;
        String[] nominalAttrNames = null;
        if (vertices.hasIntAttr()) {
            intAttrNames = Arrays.copyOf(vertices.iAttrNames,
                    vertices.getNumIntAttr());
        }
        if (vertices.hasFloatAttr()) {
            floatAttrNames = Arrays.copyOf(vertices.fAttrNames,
                    vertices.getNumFloatAttr());
        }
        if (vertices.hasNominalAttr()) {
            nominalAttrNames = Arrays.copyOf(vertices.sAttrNames,
                    vertices.getNumNominalAttr());
        }
        int numVertices = g.size() + vertices.data.size();
        mergedGraph.vertices = new DataSet(intAttrNames, floatAttrNames,
                nominalAttrNames, numVertices);
        // Fill in the vertices.
        for (int i = 0; i < size(); i++) {
            mergedGraph.vertices.addDataInstance(vertices.data.get(i));
        }
        for (int i = 0; i < g.vertices.data.size(); i++) {
            mergedGraph.vertices.addDataInstance(g.vertices.data.get(i));
        }
        // In case there is associated ID data, handle that as well.
        if ((vertices.getIdentifiers() != null)
                && (g.vertices.getIdentifiers() != null)
                && (vertices.getIdentifiers().data != null)
                && (g.vertices.getIdentifiers().data != null)
                && (!vertices.getIdentifiers().isEmpty())
                && (!g.vertices.getIdentifiers().isEmpty())) {
            DataSet idDSet = vertices.getIdentifiers();
            DataSet gIdDSet = g.vertices.getIdentifiers();
            intAttrNames = null;
            floatAttrNames = null;
            nominalAttrNames = null;
            if (idDSet.hasIntAttr()) {
                intAttrNames = Arrays.copyOf(idDSet.iAttrNames,
                        idDSet.getNumIntAttr());
            }
            if (idDSet.hasFloatAttr()) {
                floatAttrNames = Arrays.copyOf(idDSet.fAttrNames,
                        idDSet.getNumFloatAttr());
            }
            if (idDSet.hasNominalAttr()) {
                nominalAttrNames = Arrays.copyOf(idDSet.sAttrNames,
                        idDSet.getNumNominalAttr());
            }
            DataSet mergedIDDataSet =
                    new DataSet(intAttrNames, floatAttrNames,
                    nominalAttrNames, idDSet.data.size() + gIdDSet.data.size());
            for (int i = 0; i < idDSet.size(); i++) {
                mergedIDDataSet.addDataInstance(idDSet.data.get(i));
            }
            for (int i = 0; i < gIdDSet.size(); i++) {
                mergedIDDataSet.addDataInstance(gIdDSet.data.get(i));
            }
            mergedGraph.vertices.setIdentifiers(mergedIDDataSet);
        }
        // Merge the edges.
        mergedGraph.edges = new DMGraphEdge[numVertices];
        for (int i = 0; i < vertices.size(); i++) {
            mergedGraph.edges[i] = edges[i];
        }
        for (int i = 0; i < g.vertices.size(); i++) {
            mergedGraph.edges[vertices.data.size() + i] = g.edges[i];
        }
        int firstVertices = vertices.data.size();
        DMGraphEdge edge;
        for (int i = 0; i < g.vertices.data.size(); i++) {
            edge = mergedGraph.edges[firstVertices + i];
            while (edge != null) {
                edge.first += firstVertices;
                edge.second += firstVertices;
                edge = edge.next;
            }
        }
        return mergedGraph;
    }

    /**
     * @return True if empty, false otherwise.
     */
    public boolean isEmpty() {
        if ((vertices == null) || (vertices.data == null)
                || vertices.isEmpty()) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Creates a copy of the original graph.
     *
     * @return DMGraph that is the copy of the original graph.
     * @throws Exception
     */
    public DMGraph copy() throws Exception {
        DMGraph g = new DMGraph(this.networkName, this.networkDescription);
        g.vertices = vertices.copy();
        g.edges = new DMGraphEdge[edges.length];
        for (int i = 0; i < edges.length; i++) {
            g.edges[i] = edges[i].copy();
        }
        return g;
    }

    /**
     * @return Integer that is the number of vertices in the graph.
     */
    public int getNumberOfVertices() {
        return size();
    }

    /**
     * @return Integer that is the number of edges in the graph.
     */
    public int getNumberOfEdges() {
        int numEdges = 0;
        if (edges != null) {
            for (int i = 0; i < edges.length; i++) {
                DMGraphEdge edge = edges[i];
                while (edge != null) {
                    if (edge.first < edge.second) {
                        numEdges++;
                    }
                    edge = edge.next;
                }
            }
        }
        return numEdges;
    }

    /**
     * @param firstIndex Integer that is the first vertex index.
     * @param secondIndex Integer that is the second vertex index.
     * @return DMGraphEdge between the two specified vertices. If it does not
     * exist, an empty edge is returned.
     * @throws Exception
     */
    public DMGraphEdge getEdge(int firstIndex, int secondIndex)
            throws Exception {
        if (Math.max(firstIndex, secondIndex) > edges.length) {
            throw new Exception("Index out of bounds. Attempted:"
                    + firstIndex + "," + secondIndex + " |graph size: "
                    + edges.length);
        }
        DMGraphEdge edge = edges[firstIndex];
        while ((edge != null) && edge.second < secondIndex) {
            edge = edge.next;
        }
        if (edge == null) {
            return new DMGraphEdge();
        }
        if (edge.second > secondIndex) {
            return new DMGraphEdge();
        } else {
            return edge;
        }
    }

    /**
     * @param firstIndex Integer that is the first vertex index.
     * @param secondIndex Integer that is the second vertex index.
     * @return Double that is the weight of the specified edge or 0 if it does
     * not exist.
     * @throws Exception
     */
    public double gedEdgeWeight(int firstIndex, int secondIndex)
            throws Exception {
        if (Math.max(firstIndex, secondIndex) > edges.length) {
            throw new Exception("Index out of bounds. Attempted:"
                    + firstIndex + "," + secondIndex + " |graph size: "
                    + edges.length);
        }
        DMGraphEdge edge = edges[firstIndex];
        while ((edge != null) && edge.second < secondIndex) {
            edge = edge.next;
        }
        if (edge == null) {
            return 0d;
        }
        if (edge.second > secondIndex) {
            return 0d;
        } else {
            return edge.weight;
        }
    }

    /**
     * @param vertexIndex Integer that is the vertex index.
     * @return Integer that is the node degree of the specified vertex.
     * @throws Exception
     */
    public int getNodeDegree(int vertexIndex) throws Exception {
        int nodeDegree = 0;
        DMGraphEdge edge = edges[vertexIndex];
        while (edge != null) {
            nodeDegree++;
            edge = edge.next;
        }
        return nodeDegree;
    }

    /**
     * @param vertexIndex Integer that is the vertex index.
     * @return Integer that is the weighted node degree of the specified vertex.
     * @throws Exception
     */
    public double getWeightedNodeDegree(int vertexIndex) throws Exception {
        double nodeDegree = 0;
        DMGraphEdge edge = edges[vertexIndex];
        while (edge != null) {
            nodeDegree += edge.weight;
            edge = edge.next;
        }
        return nodeDegree;
    }

    /**
     * @return Integer that is the maximal node degree in the graph.
     * @throws Exception
     */
    public int getMaxNodeDegree() throws Exception {
        int maxNodeDegree = 0;
        for (int i = 0; i < edges.length; i++) {
            int nodeDegree = getNodeDegree(i);
            if (nodeDegree > maxNodeDegree) {
                maxNodeDegree = nodeDegree;
            }
        }
        return maxNodeDegree;
    }

    /**
     * @return Double that is the maximal weighted node degree in the graph.
     * @throws Exception
     */
    public double getMaxWeightedNodeDegree() throws Exception {
        double maxNodeDegree = 0.;
        for (int i = 0; i < edges.length; i++) {
            double nodeDegree = getWeightedNodeDegree(i);
            if (nodeDegree > maxNodeDegree) {
                maxNodeDegree = nodeDegree;
            }
        }
        return maxNodeDegree;
    }

    /**
     * This method inserts an edge into the graph incidence matrix.
     *
     * @param incidenceMatrix DMGraphEdge[] that is the incidence matrix for the
     * graph.
     * @param edge DMGraphEdge that is the edge to insert.
     * @throws Exception
     */
    public static void insertEdge(DMGraphEdge[] incidenceMatrix,
            DMGraphEdge edge) throws Exception {
        if (incidenceMatrix == null) {
            return;
        }
        // Look in the appropriate list.
        DMGraphEdge lookupEdge = incidenceMatrix[edge.first];
        if (lookupEdge == null) {
            incidenceMatrix[edge.first] = edge;
        } else if (lookupEdge.second > edge.second) {
            // Insert at the beginning.
            edge.next = incidenceMatrix[edge.first];
            incidenceMatrix[edge.first].previous = edge;
            incidenceMatrix[edge.first] = edge;
        } else {
            // Look for the place to insert.
            while ((lookupEdge.next != null)
                    && lookupEdge.next.second < edge.second) {
                lookupEdge = lookupEdge.next;
            }
            if (lookupEdge.next == null) {
                lookupEdge.next = edge;
                edge.previous = lookupEdge;
            } else if (lookupEdge.next.second > edge.second) {
                edge.next = lookupEdge.next;
                lookupEdge.next.previous = edge;
                edge.previous = lookupEdge;
                lookupEdge.next = edge;
            } else {
                lookupEdge.next.weight = edge.weight;
            }
        }
    }
}