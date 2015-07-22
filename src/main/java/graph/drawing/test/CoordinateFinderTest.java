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
package graph.drawing.test;

import data.representation.DataSet;
import graph.basic.DMGraph;
import graph.basic.DMGraphEdge;
import graph.basic.VertexInstance;
import graph.drawing.BarycentricCoordinateFinder;
import graph.drawing.FRCoordinateFinder;
import graph.drawing.RandomCoordinateFinder;
import java.util.Random;
import static junit.framework.Assert.assertTrue;
import junit.framework.TestCase;
import org.junit.Test;

/**
 * This class implements some basic unit tests for testing the coordinate
 * finders.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CoordinateFinderTest extends TestCase {

    /**
     * Generates a toy example for testing the coordinate finders.
     *
     * @return DMGraph that is the toy test example for testing the coordinate
     * finders.
     * @throws Exception
     */
    public static DMGraph generateToyTestData() throws Exception {
        DMGraph g = new DMGraph();
        DataSet vertexSet = new DataSet();
        vertexSet.sAttrNames = new String[1];
        vertexSet.sAttrNames[0] = "name";
        Random randa = new Random();
        for (int i = 0; i < 10; i++) {
            VertexInstance instance = new VertexInstance(vertexSet);
            instance.scale = 10 + randa.nextInt(20);
            vertexSet.addDataInstance(instance);
            instance.embedInDataset(vertexSet);
        }
        g.vertices = vertexSet;
        g.edges = new DMGraphEdge[10];
        DMGraphEdge edge;
        edge = new DMGraphEdge(0, 1, 2);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(1, 2, 1);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(2, 3, 1);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(1, 3, 1);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(3, 5, 3);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(1, 4, 2);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(1, 6, 4);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(1, 7, 1);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(1, 8, 2);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(1, 9, 3);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(6, 7, 2);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(7, 9, 1);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(6, 9, 4);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(5, 8, 5);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(3, 8, 2);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(2, 7, 1);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(0, 9, 1);
        DMGraph.insertEdge(g.edges, edge);
        edge = new DMGraphEdge(0, 8, 3);
        DMGraph.insertEdge(g.edges, edge);
        return g;
    }

    /**
     * This method tests the Fruchterman-Reingold method for calculating graph
     * vertex coordinates.
     *
     * @throws Exception
     */
    @Test
    public static void testFruchtermanReingold() throws Exception {
        DMGraph g = generateToyTestData();
        DataSet vertexSet = g.vertices;
        int frameWidth = 500;
        int frameHeight = 300;
        FRCoordinateFinder frf = new FRCoordinateFinder(g,
                frameWidth, frameHeight);
        frf.findCoordinates();
        // The method tests whether all the coordinates are set within the
        // specified frame.
        for (int i = 0; i < g.size(); i++) {
            assertTrue(((VertexInstance) vertexSet.data.get(i)).x >= 0);
            assertTrue(((VertexInstance) vertexSet.data.get(i)).x <=
                    frameWidth);
            assertTrue(((VertexInstance) vertexSet.data.get(i)).y >= 0);
            assertTrue(((VertexInstance) vertexSet.data.get(i)).y
                    <= frameHeight);
        }
    }

    /**
     * This method tests the Barycentric method for calculating graph vertex
     * coordinates.
     *
     * @throws Exception
     */
    @Test
    public static void testBarycentricMethod() throws Exception {
        DMGraph g = generateToyTestData();
        DataSet vertexSet = g.vertices;
        int frameWidth = 500;
        int frameHeight = 300;
        BarycentricCoordinateFinder bcf = new BarycentricCoordinateFinder(g,
                frameWidth, frameHeight);
        bcf.findCoordinates();
        // The method tests whether all the coordinates are set within the
        // specified frame.
        for (int i = 0; i < g.size(); i++) {
            assertTrue(((VertexInstance) vertexSet.data.get(i)).x >= 0);
            assertTrue(((VertexInstance) vertexSet.data.get(i)).x <=
                    frameWidth);
            assertTrue(((VertexInstance) vertexSet.data.get(i)).y >= 0);
            assertTrue(((VertexInstance) vertexSet.data.get(i)).y
                    <= frameHeight);
        }
    }

    /**
     * This method tests the random method for calculating graph vertex
     * coordinates.
     *
     * @throws Exception
     */
    @Test
    public static void testRandomMethod() throws Exception {
        DMGraph g = generateToyTestData();
        DataSet vertexSet = g.vertices;
        int frameWidth = 500;
        int frameHeight = 300;
        RandomCoordinateFinder rcf = new RandomCoordinateFinder(g,
                frameWidth, frameHeight);
        rcf.findCoordinates();
        // The method tests whether all the coordinates are set within the
        // specified frame.
        for (int i = 0; i < g.size(); i++) {
            assertTrue(((VertexInstance) vertexSet.data.get(i)).x >= 0);
            assertTrue(((VertexInstance) vertexSet.data.get(i)).x <=
                    frameWidth);
            assertTrue(((VertexInstance) vertexSet.data.get(i)).y >= 0);
            assertTrue(((VertexInstance) vertexSet.data.get(i)).y
                    <= frameHeight);
        }
    }
}