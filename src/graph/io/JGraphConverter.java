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
package graph.io;

import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import graph.basic.DMGraph;
import graph.basic.DMGraphEdge;
import graph.basic.VertexInstance;
import java.awt.Color;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import javax.swing.BorderFactory;
import org.jgraph.JGraph;
import org.jgraph.graph.DefaultEdge;
import org.jgraph.graph.DefaultGraphCell;
import org.jgraph.graph.DefaultGraphModel;
import org.jgraph.graph.GraphConstants;
import org.jgraph.graph.GraphModel;

/**
 * This class contains the methods for generating an associated JGraph object
 * for a DMGraph object, for display purposes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class JGraphConverter {

    /**
     * Converts a DMGraph object to a JGraph object. This method assumes that
     * the scale and coordinates are given separately, though in fact they can
     * also be given within the VertexInstances in the graph.
     *
     * @param g DMGraph to be converted.
     * @param coordinates double[][] representing the vertex coordinates.
     * @param scale double[] representing vertex scales.
     * @param featureType Integer representing the feature type for display.
     * @param featureIndex Integer representing the feature index to display.
     * @return JGraph that corresponds to the provided DMGraph.
     * @throws Exception
     */
    public static JGraph dmGraphToJGraph(DMGraph g, double[][] coordinates,
            double[] scale, int featureType, int featureIndex)
            throws Exception {
        DefaultGraphModel dgm = new DefaultGraphModel();
        JGraph jg = new JGraph(dgm);
        jg.setCloneable(true);
        jg.setInvokesStopCellEditing(true);
        jg.setJumpToDefaultPort(true);
        DefaultGraphCell[] cells =
                new DefaultGraphCell[g.vertices.size() + g.getNumberOfEdges()];
        String cellName = "";
        for (int vertexIndex = 0; vertexIndex < g.size(); vertexIndex++) {
            // First we extract the cell name;
            switch (featureType) {
                case (DataMineConstants.NOMINAL): {
                    cellName = g.vertices.data.get(vertexIndex).sAttr[
                            featureIndex];
                    break;
                }
                case (DataMineConstants.INTEGER): {
                    cellName = (new Integer(g.vertices.data.get(
                            vertexIndex).iAttr[featureIndex])).toString();
                    break;
                }
                case (DataMineConstants.FLOAT): {
                    cellName = (new Float(g.vertices.data.get(
                            vertexIndex).fAttr[featureIndex])).toString();
                    break;
                }
            }
            // We create the appropriate cell.
            cells[vertexIndex] = createCell(cellName,
                    coordinates[vertexIndex][0], coordinates[vertexIndex][1],
                    scale[vertexIndex] * 10);
            ((VertexInstance) (g.vertices.data.get(vertexIndex))).jgVertex =
                    cells[vertexIndex];
        }
        for (int vertexIndex = 0; vertexIndex < g.size(); vertexIndex++) {
            DMGraphEdge edge = g.edges[vertexIndex];
            while (edge != null) {
                if (edge.first < edge.second) {
                    cells[g.vertices.size() + vertexIndex] =
                            createWeightedEdge(cells[edge.first],
                            cells[edge.second], edge.weight);
                }
                edge = edge.next;
            }
        }
        insertCellArray(jg, cells);
        return jg;
    }

    /**
     * Converts a DMGraph object to a JGraph object.
     *
     * @param g DMGraph to be converted.
     * @param featureType Integer representing the feature type for display.
     * @param featureIndex Integer representing the feature index to display.
     * @return JGraph that corresponds to the provided DMGraph.
     * @throws Exception
     */
    public static JGraph dmGraphToJGraph(DMGraph g, int featureType,
            int featureIndex) throws Exception {
        DefaultGraphModel dgm = new DefaultGraphModel();
        JGraph jg = new JGraph(dgm);
        jg.setCloneable(true);
        jg.setInvokesStopCellEditing(true);
        jg.setJumpToDefaultPort(true);
        DefaultGraphCell[] cells = new DefaultGraphCell[g.vertices.size()
                + g.getNumberOfEdges()];
        String cellName = null;
        for (int vertexIndex = 0; vertexIndex < g.size(); vertexIndex++) {
            // First we extract the cell name;
            switch (featureType) {
                case (DataMineConstants.NOMINAL): {
                    cellName = g.vertices.data.get(vertexIndex).sAttr[
                            featureIndex];
                    break;
                }
                case (DataMineConstants.INTEGER): {
                    cellName = (new Integer(g.vertices.data.get(
                            vertexIndex).iAttr[featureIndex])).toString();
                    break;
                }
                case (DataMineConstants.FLOAT): {
                    cellName = (new Float(g.vertices.data.get(
                            vertexIndex).fAttr[featureIndex])).toString();
                    break;
                }
            }
            // We create the appropriate cell.
            cells[vertexIndex] = createCell(cellName,
                    ((VertexInstance) (g.vertices.data.get(vertexIndex))).x,
                    ((VertexInstance) (g.vertices.data.get(vertexIndex))).y,
                    ((VertexInstance) (g.vertices.data.get(vertexIndex))).scale
                    * 10);
            ((VertexInstance) (g.vertices.data.get(vertexIndex))).jgVertex =
                    cells[vertexIndex];
        }
        // Now handle the edges.
        int edgeCounter = 0;
        for (int i = 0; i < g.size(); i++) {
            DMGraphEdge edge = g.edges[i];
            while (edge != null) {
                if (edge.first < edge.second) {
                    cells[g.vertices.size() + edgeCounter++] =
                            createWeightedEdge(cells[edge.first],
                            cells[edge.second], edge.weight);
                }
                edge = edge.next;
            }
        }
        insertCellArray(jg, cells);
        g.visGraph = jg;
        return jg;
    }

    /**
     * Creates a DMGraph object from a JGraph object.
     *
     * @param jg JGraph object to infer the DMGraph object from.
     * @param networkName String that is the network name.
     * @param networkDescription String that is the network description.
     * @return DMGraph that corresponds to the specified JGraph.
     * @throws Exception
     */
    public static DMGraph jgraphToDMGraph(JGraph jg, String networkName,
            String networkDescription) throws Exception {
        DMGraph g = jgraphToDMGraph(jg);
        g.networkName = networkName;
        g.networkDescription = networkDescription;
        return g;
    }

    /**
     * Creates a DMGraph object from a JGraph object.
     *
     * @param jg JGraph object to infer the DMGraph object from.
     * @return DMGraph that corresponds to the specified JGraph.
     * @throws Exception
     */
    public static DMGraph jgraphToDMGraph(JGraph jg) throws Exception {
        GraphModel gm = jg.getModel();
        Object[] cells = DefaultGraphModel.getAll(gm);
        DefaultGraphCell[] vertices = new DefaultGraphCell[cells.length];
        int numVertices = 0;
        int numPorts = 0;
        for (int cellIndex = 0; cellIndex < cells.length; cellIndex++) {
            if (!((cells[cellIndex] instanceof org.jgraph.graph.DefaultEdge)
                    || (cells[cellIndex] instanceof
                    org.jgraph.graph.DefaultPort))) {
                vertices[numVertices++] = (DefaultGraphCell) (cells[cellIndex]);
            }
            if (cells[cellIndex] instanceof org.jgraph.graph.DefaultPort) {
                numPorts++;
            }
        }
        DefaultEdge[] edges = new DefaultEdge[cells.length - numVertices
                - numPorts];
        int numEdges = 0;
        for (int cellIndex = 0; cellIndex < cells.length; cellIndex++) {
            if (cells[cellIndex] instanceof org.jgraph.graph.DefaultEdge) {
                edges[numEdges++] = (DefaultEdge) (cells[cellIndex]);
            }
        }
        DMGraph g = new DMGraph();
        String[] nominalNames = new String[1];
        nominalNames[0] = "name";
        DataSet graphVertices = new DataSet(null, null, nominalNames);
        graphVertices.data = new ArrayList<>(numVertices);
        for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
            VertexInstance vi = new VertexInstance(graphVertices);
            String name = (String) (gm.getValue(vertices[vertexIndex]));
            Rectangle2D.Double boundingRect = (Rectangle2D.Double) (
                    jg.getCellBounds(vertices[vertexIndex]));
            vi.x = boundingRect.getX();
            vi.y = boundingRect.getY();
            vi.scale = boundingRect.getHeight() / 10;
            if (vi.sAttr == null) {
                vi.sAttr = new String[1];
            }
            vi.sAttr[0] = name;
            vi.jgVertex = vertices[vertexIndex];
            graphVertices.addDataInstance(vi);
        }
        g.vertices = graphVertices;
        DMGraphEdge[] graphEdges = new DMGraphEdge[numEdges];
        g.edges = graphEdges;
        for (int edgeIndex = 0; edgeIndex < numEdges; edgeIndex++) {
            DMGraphEdge edge = new DMGraphEdge();
            Object firstOb = DefaultGraphModel.getSourceVertex(gm,
                    edges[edgeIndex]);
            Object secondOb = DefaultGraphModel.getTargetVertex(gm,
                    edges[edgeIndex]);
            int first = -1;
            int second = -1;
            int counter = 0;
            while ((counter < numVertices) && ((second == -1)
                    || (first == -1))) {
                if (vertices[counter] == firstOb) {
                    first = counter;
                }
                if (vertices[counter] == secondOb) {
                    second = counter;
                }
                counter++;
            }
            if (second < first) {
                int exchange = first;
                first = second;
                second = exchange;
            }
            edge.first = first;
            edge.second = second;
            edge.weight = (new Double((String) (
                    gm.getValue(edges[edgeIndex])))).doubleValue();
            DMGraph.insertEdge(graphEdges, edge);
        }
        g.visGraph = jg;
        return g;
    }

    /**
     * Inserts an array of cells into a JGraph object.
     *
     * @param jg JGraph object to insert the cells to.
     * @param cells DefaultGraphCell[] of cells to insert into the JGraph
     * object.
     * @throws Exception
     */
    public static void insertCellArray(JGraph jg, DefaultGraphCell[] cells)
            throws Exception {
        jg.getGraphLayoutCache().insert(cells);
    }

    /**
     * Sets the coordinates for a cell in JGraph.
     *
     * @param jg JGraph object that is the visual context.
     * @param cell DefaultGraphCell that is to be updated.
     * @param x Double that is the X coordinate.
     * @param y Double that is the Y coordinate.
     * @param scale Double that is the scale information.
     * @throws Exception
     */
    public static void setCellCoordinates(JGraph jg, DefaultGraphCell cell,
            double x, double y, double scale) throws Exception {
        try {
            Map attributes = cell.getAttributes();
            GraphConstants.setBounds(attributes,
                    new Rectangle2D.Double(x, y, scale, scale));
            Map attributeMap = new HashMap();
            attributeMap.put(cell, attributes);
            GraphModel model = jg.getModel();
            model.edit(attributeMap, null, null, null);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * Creates a new JGraph cell with the specified rendering information.
     *
     * @param name String that is the cell name.
     * @param x Double that is the X coordinate.
     * @param y Double that is the Y coordinate.
     * @param scale Double that is the scale information.
     * @param color Color of the future graph cell.
     * @return DMGraph cell generated with the specified rendering information.
     * @throws Exception
     */
    public static DefaultGraphCell createCell(String name, double x, double y,
            double scale, Color color) throws Exception {
        DefaultGraphCell cell = new DefaultGraphCell(name);
        GraphConstants.setBounds(cell.getAttributes(),
                new Rectangle2D.Double(x, y, scale, scale));
        GraphConstants.setBorder(cell.getAttributes(),
                BorderFactory.createRaisedBevelBorder());
        GraphConstants.setOpaque(cell.getAttributes(), true);
        cell.addPort(new Point2D.Double(0, 0));
        if (color != null) {
            GraphConstants.setGradientColor(cell.getAttributes(), color);
        } else {
            GraphConstants.setGradientColor(cell.getAttributes(), Color.BLUE);
        }
        return cell;
    }

    /**
     * Creates a new JGraph cell with the specified rendering information.
     *
     * @param name String that is the cell name.
     * @param x Double that is the X coordinate.
     * @param y Double that is the Y coordinate.
     * @param scale Double that is the scale information.
     * @return DMGraph cell generated with the specified rendering information.
     * @throws Exception
     */
    public static DefaultGraphCell createCell(String name, double x, double y,
            double scale) throws Exception {
        DefaultGraphCell cell = new DefaultGraphCell(name);
        GraphConstants.setBounds(cell.getAttributes(),
                new Rectangle2D.Double(x, y, scale, scale));
        GraphConstants.setBorder(cell.getAttributes(),
                BorderFactory.createRaisedBevelBorder());
        GraphConstants.setOpaque(cell.getAttributes(), true);
        GraphConstants.setGradientColor(cell.getAttributes(), Color.BLUE);
        cell.addPort(new Point2D.Double(0, 0));
        return cell;
    }

    /**
     * Updates the coordinates in the visual JGraph model based on the
     * associated DMGraph data model.
     *
     * @param g DMGraph that is to be updated.
     * @throws Exception
     */
    public static void updateJGCoordinates(DMGraph g) throws Exception {
        if (g.isEmpty()) {
            return;
        }
        if (g.visGraph == null) {
            return;
        }
        for (int i = 0; i < g.vertices.data.size(); i++) {
            if (((VertexInstance) (g.vertices.data.get(i))).jgVertex != null) {
                setCellCoordinates(g.visGraph, ((VertexInstance) (
                        g.vertices.data.get(i))).jgVertex,
                        ((VertexInstance) (g.vertices.data.get(i))).x,
                        ((VertexInstance) (g.vertices.data.get(i))).y,
                        ((VertexInstance) (g.vertices.data.get(i))).scale);
            }
        }
    }

    /**
     * Creates a new edge in JGraph.
     *
     * @param source DefaultGraphCell that is the source for the edge.
     * @param target DefaultGraphCell that is the target for the edge.
     * @return DefaultGraphCell that was generated to connect the source with
     * the target.
     * @throws Exception
     */
    public static DefaultGraphCell createEdge(DefaultGraphCell source,
            DefaultGraphCell target) throws Exception {
        DefaultEdge edge = new DefaultEdge();
        source.addPort();
        edge.setSource(source.getChildAt(source.getChildCount() - 1));
        target.addPort();
        edge.setTarget(target.getChildAt(target.getChildCount() - 1));
        return edge;
    }

    /**
     * Creates a new weighted edge in JGraph.
     *
     * @param source DefaultGraphCell that is the source for the edge.
     * @param target DefaultGraphCell that is the target for the edge.
     * @param weight Double that is the weight of the edge.
     * @return DefaultGraphCell that was generated to connect the source with
     * the target, with the specified weight.
     * @throws Exception
     */
    public static DefaultGraphCell createWeightedEdge(DefaultGraphCell source,
            DefaultGraphCell target, double weight) throws Exception {
        DefaultEdge edge = new DefaultEdge((new Double(weight)).toString());
        source.addPort();
        edge.setSource(source.getChildAt(source.getChildCount() - 1));
        target.addPort();
        edge.setTarget(target.getChildAt(target.getChildCount() - 1));
        return edge;
    }
}