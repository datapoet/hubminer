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
package graph.drawing;

import graph.basic.DMGraph;
import graph.basic.DMGraphEdge;
import graph.basic.VertexInstance;
import graph.basic.VertexScaleComparator;
import graph.io.JGraphConverter;
import java.awt.geom.Point2D;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 * This class implements the method for calculating the barycentric graph vertex
 * coordinates.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BarycentricCoordinateFinder implements CoordinateFinderInterface {

    // Width and height of the target canvas.
    private int width;
    private int height;
    // Number of sides of the target polygon.
    private int polygonNumSides = -1;
    // Iteration error for convergence checks.
    private double totalError = 0.;
    private static final double CONVERGENCE_THRESHOLD = 2.7;
    private static final int MAX_ITER = 100;
    // The graph that we are trying to find the coordinates for.
    private DMGraph g = null;
    // Boolean flag indicating whether the coordinate finder is currently
    // running.
    private boolean isRunning = true;
    // Estimated current progress of coordinate calculations.
    private double progress = 0.;
    // Boolean flag indicating whether to auto-update the associated JGraph.
    private boolean updateJG = false;

    @Override
    public void setAutoJGUpdate(boolean updateJG) {
        this.updateJG = updateJG;
    }

    @Override
    public double getProgress() {
        return progress;
    }

    @Override
    public void stop() {
        isRunning = false;
    }

    /**
     * If the algorithm has finished calculations, interrupt the execution of
     * the thread.
     *
     * @throws Exception
     */
    private void check() throws Exception {
        if (!isRunning) {
            throw new InterruptedException();
        }
    }

    /**
     * Initialization.
     *
     * @param g DMGraph to find the vertex coordinates for.
     * @param width Integer that is the target canvas width.
     * @param height Integer that is the target canvas height.
     */
    public BarycentricCoordinateFinder(DMGraph g, int width, int height) {
        this.g = g;
        this.width = width;
        this.height = height;
    }

    /**
     * Initialization.
     *
     * @param g DMGraph to find the vertex coordinates for.
     * @param width Integer that is the target canvas width.
     * @param height Integer that is the target canvas height.
     * @param polygonNumSides Integer that is the number of sides of the target
     * polygon.
     */
    public BarycentricCoordinateFinder(DMGraph g, int width, int height,
            int polygonNumSides) {
        this.g = g;
        this.width = width;
        this.height = height;
        this.polygonNumSides = polygonNumSides;
    }

    @Override
    public void run() {
        try {
            findCoordinates();
        } catch (Exception e) {
        }
    }

    @Override
    public void findCoordinates() throws Exception {
        int numVertices = g.getNumberOfVertices();
        if (polygonNumSides < 3) {
            // Determine the suitable polygon complexity.
            int candidateNumSides = numVertices / 10;
            polygonNumSides = Math.min(Math.max(3, candidateNumSides), 30);
        }
        HashMap<VertexInstance, Integer> vertexIndexMap =
                new HashMap<>(numVertices * 2);
        // Order the vertices, find the most important ones and place them at
        // the polygon vertices.
        VertexInstance[] sortedVertices = new VertexInstance[numVertices];
        double maxVertexScale = 0.;
        // First just fill up the array in linear order.
        for (int i = 0; i < numVertices; i++) {
            sortedVertices[i] = (VertexInstance) (g.vertices.data.get(i));
            vertexIndexMap.put((VertexInstance) (g.vertices.data.get(i)), i);
            if (sortedVertices[i].scale > maxVertexScale) {
                maxVertexScale = sortedVertices[i].scale;
            }
        }
        VertexScaleComparator vsc = new VertexScaleComparator(
                VertexScaleComparator.DESCENDING);
        // Now do the actual sorting.
        Arrays.sort(sortedVertices, vsc);
        // Adjust the width and height of the surrounding ellipse, in order for
        // the nodes to stay in the frame when being drawn.
        double wEllipse = width - (2 * maxVertexScale);
        double hEllipse = height - (2 * maxVertexScale);
        for (int i = 0; i < polygonNumSides; i++) {
            double angle = 2. * Math.PI * (i + 1) / (double) (polygonNumSides);
            sortedVertices[i].x = wEllipse / 2 + Math.cos(angle) * wEllipse / 2;
            sortedVertices[i].y = hEllipse / 2 + Math.sin(angle) * hEllipse / 2;
        }
        HashMap<Integer, Integer> selectedIndexMap =
                new HashMap<>(polygonNumSides * 2);
        for (int i = 0; i < polygonNumSides; i++) {
            selectedIndexMap.put(vertexIndexMap.get(sortedVertices[i]), i);
        }
        // Determine the node degree for each vertex.
        int[] vertexDegree = new int[numVertices];
        for (int i = 0; i < numVertices; i++) {
            vertexDegree[i] = g.getNodeDegree(i);
        }
        // Randomize the starting positions.
        Random randa = new Random();
        for (int i = polygonNumSides; i < numVertices; i++) {
            sortedVertices[i].x = maxVertexScale + (randa.nextInt(
                    (int) wEllipse)
                    % (int) (width - maxVertexScale));
            sortedVertices[i].y = maxVertexScale + (randa.nextInt(
                    (int) hEllipse)
                    % (int) (height - maxVertexScale));
        }
        Point2D.Double[] nextPosition = new Point2D.Double[numVertices];
        for (int i = 0; i < numVertices; i++) {
            nextPosition[i] = new Point2D.Double();
            nextPosition[i].x = 0;
            nextPosition[i].y = 0;
        }
        int iteration = 0;
        do {
            iteration++;
            // Checking to see if the thread was signalled to stop in the
            // meantime.
            check();
            totalError = 0.;
            // First just sum things up.
            for (int i = 0; i < numVertices; i++) {
                DMGraphEdge edge = g.edges[i];
                while (edge != null) {
                    if (edge.first < edge.second) {
                        if (!selectedIndexMap.containsKey(edge.first)) {
                            nextPosition[edge.first].x +=
                                    ((VertexInstance) (g.vertices.data.get(
                                    edge.second))).x;
                            nextPosition[edge.first].y +=
                                    ((VertexInstance) (g.vertices.data.get(
                                    edge.second))).y;
                        }
                        if (!selectedIndexMap.containsKey(edge.second)) {
                            nextPosition[edge.second].x +=
                                    ((VertexInstance) (g.vertices.data.get(
                                    edge.first))).x;
                            nextPosition[edge.second].y +=
                                    ((VertexInstance) (g.vertices.data.get(
                                    edge.first))).y;
                        }
                    }
                    edge = edge.next;
                }
            }
            // Calculate the averages.
            for (int i = 0; i < numVertices; i++) {
                if (!selectedIndexMap.containsKey(i)) {
                    if (vertexDegree[i] > 0) {
                        nextPosition[i].x /= vertexDegree[i];
                        nextPosition[i].y /= vertexDegree[i];
                    } else {
                        nextPosition[i].x =
                                ((VertexInstance) (g.vertices.data.get(i))).x;
                        nextPosition[i].y =
                                ((VertexInstance) (g.vertices.data.get(i))).y;
                    }
                    nextPosition[i].x = maxVertexScale
                            + (nextPosition[i].x % wEllipse);
                    nextPosition[i].y = maxVertexScale
                            + (nextPosition[i].y % hEllipse);
                }
            }
            // Calculate the total iteration error.
            // Checking to see if the thread was signalled to stop in the
            // meantime.
            check();
            for (int i = 0; i < numVertices; i++) {
                if (!selectedIndexMap.containsKey(i)) {
                    totalError += Math.abs(nextPosition[i].x
                            - ((VertexInstance) (g.vertices.data.get(i))).x)
                            + Math.abs(nextPosition[i].y
                            - ((VertexInstance) (g.vertices.data.get(i))).y);
                    ((VertexInstance) (g.vertices.data.get(i))).x =
                            nextPosition[i].x;
                    ((VertexInstance) (g.vertices.data.get(i))).y =
                            nextPosition[i].y;
                    if (updateJG) {
                        if (((VertexInstance) (g.vertices.data.get(i))).
                                jgVertex != null) {
                            JGraphConverter.setCellCoordinates(
                                    g.visGraph,
                                    ((VertexInstance) (g.vertices.data.get(
                                    i))).jgVertex,
                                    nextPosition[i].x,
                                    nextPosition[i].y,
                                    ((VertexInstance) (g.vertices.data.get(
                                    i))).scale);
                        }
                    }
                    nextPosition[i].x = 0.;
                    nextPosition[i].y = 0.;
                }
            }
            // Not really accurate.
            progress = Math.min(progress + 0.03, 1.);
        } while (!(totalError < CONVERGENCE_THRESHOLD
                * (numVertices - polygonNumSides)) && iteration < MAX_ITER);
        progress = 1.;
        if (!updateJG) {
            progress = 0.99;
            JGraphConverter.updateJGCoordinates(g);
            progress = 1.;
        }
    }
}