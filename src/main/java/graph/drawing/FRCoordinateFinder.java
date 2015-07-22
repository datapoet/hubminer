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

import data.representation.util.DataMineConstants;
import graph.basic.DMGraph;
import graph.basic.DMGraphEdge;
import graph.basic.VertexInstance;
import graph.io.JGraphConverter;
import java.awt.geom.Point2D;
import java.util.Random;

/**
 * This class implements the Fruchterman-Reingold algorithm for graph
 * visualization. The attractive forces are calculated for the adjacent vertices
 * and the repulsive forces are calculated for all the vertices. There is also a
 * notion of temperature, as in simulated annealing, controlling the convergence
 * of the process.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class FRCoordinateFinder implements CoordinateFinderInterface {

    public static final int DEFAULT_ITERATIONS = 50;
    public static final double FINAL_TEMPERATURE = 3.;
    private int numIterations = DEFAULT_ITERATIONS;
    //The square root of the average area per vertex. This symbolizes the edge
    // length of the average covering square for a vertex in the graph.
    private double areaRoot;
    private double temperature;
    private double cooling_factor;
    private int width; // Frame width.
    private int height; // Frame height.
    private double area; // Frame area.
    // Stores the current delta for each vertex based on the attractive and
    // repulsive forces.
    private Point2D.Double[] vertexDispositions;
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
     * Reduce the system temperature.
     */
    private void iterationCoolDown() {
        temperature = temperature * cooling_factor;
    }

    /**
     * Calculates the attractive force.
     *
     * @param distance Double value that is the distance to calculate the
     * attractive force for.
     * @return Double that is the attractive force.
     * @throws Exception
     */
    private double attraction(double distance) throws Exception {
        return ((distance * distance) / areaRoot);
    }

    /**
     * Calculates the repulsive force.
     *
     * @param distance Double value that is the distance to calculate the
     * repulsive force for.
     * @return Double that is the repulsive force.
     * @throws Exception
     */
    private double repulsion(double distance) throws Exception {
        return ((areaRoot * areaRoot) / distance);
    }

    /**
     * Calculates the norm of the vector.
     *
     * @param vect Point2D.Double representing a vector in 2D.
     * @return Double that is the norm of the represented vector.
     */
    private static double modulo(Point2D.Double vect) {
        return Math.pow(vect.x * vect.x + vect.y * vect.y, 0.5);
    }

    /**
     * Initialization.
     *
     * @param g DMGraph to calculate the coordinates for.
     * @param width Integer that is the frame width.
     * @param height Integer that is the frame height.
     */
    public FRCoordinateFinder(DMGraph g, int width, int height) {
        this.g = g;
        this.width = width;
        this.height = height;
        area = width * height;
        // Initialize temperature.
        temperature = Math.max((((double) width) / 2), (((double) height) / 2));
        cooling_factor = Math.pow((FINAL_TEMPERATURE / temperature),
                1 - numIterations);
    }

    /**
     * Initialization.
     *
     * @param g DMGraph to calculate the coordinates for.
     * @param width Integer that is the frame width.
     * @param height Integer that is the frame height.
     * @param numIterations Integer that is the number of iterations to run.
     */
    public FRCoordinateFinder(DMGraph g, int width, int height,
            int numIterations) {
        this.g = g;
        this.width = width;
        this.height = height;
        area = width * height;
        this.numIterations = Math.max(numIterations, 10);
        // Initialize temperature.
        temperature = Math.max((((double) width) / 2), (((double) height) / 2));
        cooling_factor = Math.pow((FINAL_TEMPERATURE / temperature),
                1 - numIterations);
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
        areaRoot = Math.pow((area / numVertices), 0.5);
        vertexDispositions = new Point2D.Double[numVertices];
        Random randa = new Random();
        for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
            vertexDispositions[vertexIndex] = new Point2D.Double(0., 0.);
            ((VertexInstance) (g.vertices.data.get(vertexIndex))).x =
                    randa.nextDouble() * width;
            ((VertexInstance) (g.vertices.data.get(vertexIndex))).y =
                    randa.nextDouble() * height;
        }
        Point2D.Double delta = new Point2D.Double(0., 0.);
        double modulo;
        for (int iterationIndex = 0; iterationIndex < numIterations;
                iterationIndex++) {
            // Checking to see if the thread was signalled to stop in the
            // meantime.
            check();
            // First calculate all the repulsive forces.
            for (int firstVertexIndex = 0; firstVertexIndex < numVertices;
                    firstVertexIndex++) {
                for (int secondVertexIndex = firstVertexIndex + 1;
                        secondVertexIndex < numVertices; secondVertexIndex++) {
                    delta.x = ((VertexInstance) (g.vertices.data.get(
                            firstVertexIndex))).x
                            - ((VertexInstance) (g.vertices.data.get(
                            secondVertexIndex))).x;
                    delta.y = ((VertexInstance) (g.vertices.data.get(
                            firstVertexIndex))).y
                            - ((VertexInstance) (g.vertices.data.get(
                            secondVertexIndex))).y;
                    modulo = modulo(delta);
                    if (DataMineConstants.isAcceptableDouble(modulo)) {
                        vertexDispositions[firstVertexIndex].x +=
                                ((delta.x / modulo) * repulsion(modulo));
                        vertexDispositions[firstVertexIndex].y +=
                                ((delta.y / modulo) * repulsion(modulo));
                        vertexDispositions[secondVertexIndex].x -=
                                ((delta.x / modulo) * repulsion(modulo));
                        vertexDispositions[secondVertexIndex].y -=
                                ((delta.y / modulo) * repulsion(modulo));
                    }
                }
            }
            // Checking to see if the thread was signalled to stop in the
            // meantime.
            check();
            // Calculate all the attractive forces.
            for (int vertexIndex = 0; vertexIndex < numVertices;
                    vertexIndex++) {
                DMGraphEdge edge = g.edges[vertexIndex];
                while ((edge != null) && (edge.first < edge.second)) {
                    delta.x = ((VertexInstance) (g.vertices.data.get(
                            edge.first))).x
                            - ((VertexInstance) (g.vertices.data.get(
                            edge.second))).x;
                    delta.y = ((VertexInstance) (g.vertices.data.get(
                            edge.first))).y
                            - ((VertexInstance) (g.vertices.data.get(
                            edge.second))).y;
                    modulo = modulo(delta);
                    if (DataMineConstants.isAcceptableDouble(modulo)) {
                        vertexDispositions[edge.first].x -=
                                ((delta.x / modulo) * attraction(modulo));
                        vertexDispositions[edge.first].y -=
                                ((delta.y / modulo) * attraction(modulo));
                        vertexDispositions[edge.second].x +=
                                ((delta.x / modulo) * attraction(modulo));
                        vertexDispositions[edge.second].y +=
                                ((delta.y / modulo) * attraction(modulo));
                    }
                    edge = edge.next;
                }
            }
            // Checking to see if the thread was signalled to stop in the
            // meantime.
            check();
            // Limit the dispositions according to the temperature and make sure
            // that nothing gets outside of the frame.
            double vXNext;
            double vYNext;
            for (int vertexIndex = 0; vertexIndex < numVertices;
                    vertexIndex++) {
                modulo = modulo(vertexDispositions[vertexIndex]);
                // Limit the deltas by the temperature.
                if (modulo > 0.01) {
                    vXNext = ((VertexInstance) (g.vertices.data.get(
                            vertexIndex))).x
                            + ((vertexDispositions[vertexIndex].x / modulo)
                            * Math.min(modulo, temperature));
                    vYNext = ((VertexInstance) (g.vertices.data.get(
                            vertexIndex))).y
                            + ((vertexDispositions[vertexIndex].y / modulo)
                            * Math.min(modulo, temperature));
                } else {
                    vXNext = ((VertexInstance) (g.vertices.data.get(
                            vertexIndex))).x;
                    vYNext = ((VertexInstance) (g.vertices.data.get(
                            vertexIndex))).y;
                }
                // Ensure frame consistency.
                double scale = ((VertexInstance) (g.vertices.data.get(
                        vertexIndex))).scale;
                ((VertexInstance) (g.vertices.data.get(vertexIndex))).x =
                        Math.min((((double) width) - 1. - scale),
                        Math.max(vXNext, 1. + scale));
                ((VertexInstance) (g.vertices.data.get(vertexIndex))).y =
                        Math.min((((double) height) - 1. - scale),
                        Math.max(vYNext, 1. + scale));
                // If the JGraph auto-update mode is on, update the associated
                // JGraph object as well.
                if (updateJG) {
                    if (((VertexInstance) (g.vertices.data.get(
                            vertexIndex))).jgVertex != null) {
                        JGraphConverter.setCellCoordinates(g.visGraph,
                                ((VertexInstance) (g.vertices.data.get(
                                vertexIndex))).jgVertex, ((VertexInstance)
                                (g.vertices.data.get(vertexIndex))).x,
                                ((VertexInstance) (g.vertices.data.get(
                                vertexIndex))).y, scale);
                    }
                }
            }
            // Checking to see if the thread was signalled to stop in the
            // meantime.
            check();
            // Rest the dispositions before the next iteration.
            for (int vertexIndex = 0; vertexIndex < numVertices;
                    vertexIndex++) {
                vertexDispositions[vertexIndex].x = 0.;
                vertexDispositions[vertexIndex].y = 0.;
            }
            // Update the temperature of the system.
            iterationCoolDown();
            progress = (((double) (iterationIndex + 1)) /
                    (double) numIterations);
        }
        if (!updateJG) {
            progress = 0.99;
            JGraphConverter.updateJGCoordinates(g);
            progress = 1.;
        }
    }
}