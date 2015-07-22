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
package gui.images;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.neighbors.hubness.BucketedOccDistributionGetter;
import data.neighbors.hubness.HubOrphanRegularPercentagesCalculator;
import data.neighbors.hubness.HubnessExtremesGrabber;
import data.neighbors.hubness.HubnessSkewAndKurtosisExplorer;
import data.neighbors.hubness.HubnessAboveThresholdExplorer;
import data.neighbors.hubness.KNeighborEntropyExplorer;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.images.color.ColorHistogramVector;
import data.representation.images.sift.LFeatRepresentation;
import distances.primary.CombinedMetric;
import distances.secondary.LocalScalingCalculator;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import draw.basic.BoxBlur;
import draw.basic.ColorPalette;
import draw.basic.ScreenImage;
import draw.charts.PieRenderer;
import edu.uci.ics.jung.algorithms.layout.CircleLayout;
import edu.uci.ics.jung.algorithms.layout.Layout;
import edu.uci.ics.jung.graph.DirectedGraph;
import edu.uci.ics.jung.graph.DirectedSparseMultigraph;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.util.Context;
import edu.uci.ics.jung.graph.util.EdgeType;
import edu.uci.ics.jung.visualization.VisualizationViewer;
import edu.uci.ics.jung.visualization.control.PickingGraphMousePlugin;
import edu.uci.ics.jung.visualization.control.PluggableGraphMouse;
import edu.uci.ics.jung.visualization.picking.PickedState;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.images.ConvertJPGToPGM;
import ioformat.images.SiftUtil;
import java.awt.Color;
import java.awt.Component;
import java.awt.ComponentOrientation;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Vector;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableModel;
import learning.supervised.Classifier;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import learning.supervised.methods.knn.AKNN;
import learning.supervised.methods.knn.DWHFNN;
import learning.supervised.methods.knn.FNN;
import learning.supervised.methods.knn.HwKNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import data.neighbors.NSFUserInterface;
import images.mining.codebook.GenericCodeBook;
import ioformat.images.OpenCVFeatureIO;
import java.awt.HeadlessException;
import java.io.IOException;
import learning.supervised.methods.knn.NWKNN;
import mdsj.MDSJ;
import org.apache.commons.collections15.Predicate;
import org.apache.commons.collections15.Transformer;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PiePlot3D;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.StackedBarRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.util.Rotation;
import util.ArrayUtil;
import util.AuxSort;
import util.BasicMathUtil;

/**
 * This GUI was made with the intention of helping with analyzing the hubness of
 * the data and image data in particular, in terms of the skewed distribution of
 * implied relevance stemming from a particular choice of metric and feature
 * representation. It offers the users a choice between many standard primary
 * metrics and a set of state-of-the-art secondary metrics for hubness-aware
 * metric learning in order to improve the semantic consistency of between-image
 * similarities. It is composed of many visualization components for different
 * types of data overviews, with an emphasis on kNN set structure and hub
 * analysis. When using it, make sure to cite the following paper: Image Hub
 * Explorer: Evaluating Representations and Metrics for Content-based Image
 * Retrieval and Object Recognition, Nenad Tomasev and Dunja Mladenic, 2013,
 * ECML/PKDD conference.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImageHubExplorer extends javax.swing.JFrame {

    // The workspace directory.
    private File workspace;
    // The current working directory.
    private File currentDirectory;
    private static final int PRIMARY_METRIC = 0;
    private static final int SECONDARY_METRIC = 1;
    private ImageHubExplorer frameReference = this;
    // Codebook data structures for visual word analysis.
    private GenericCodeBook codebook = null;
    private double[][] codebookProfiles = null;
    private float[] codebookGoodness = null;
    // Whether to also load the secondary distances from the disk (if available)
    // or to calculate them inside Image Hub Explorer. If the latter is selected
    // then the metric object is also available for external searches.
    private boolean secondaryLoadFlag = false;
    // This flag informs the methods that the system is currently calculating
    // something, so the user is prevented from executing certain actions.
    private volatile boolean busyCalculating = false;
    // Image data that is being analyzed.
    private BufferedImage[] images;
    // Thumbnails of the images for visualization.
    private ArrayList<BufferedImage> thumbnails;
    // Reverse neighbor sets for all the images for all the neighborhood sizes.
    private ArrayList<Integer>[][] rnnSetsAllK;
    // Neighbor occurrence profiles for all neighborhood sizes.
    private float[][][] occurrenceProfilesAllK;
    // Quantized data representation, if a representation is available. It is
    // possible to operate based on the loaded distances alone, if that is
    // necessary in the context of analysis.
    private DataSet quantizedRepresentation;
    // Number of classes in the data.
    private int numClasses;
    // Colors to use for different classes.
    private Color[] classColors;
    // Names of different classes.
    private String[] classNames;
    // Files containing primary and secondary distance matrices.
    private File primaryDMatFile;
    private File secondaryDMatFile;
    // Primary distance matrix.
    private float[][] distMatrixPrimary;
    // kNN sets in the primary distance.
    private NeighborSetFinder nsfPrimary;
    // Secondary distance matrix.
    private float[][] distMatrixSecondary;
    // kNN sets in the secondary distance.
    private NeighborSetFinder nsfSecondary;
    // CombinedMetric objects for distance calculations.
    private CombinedMetric primaryCMet = null;
    private CombinedMetric secondaryCMet = null;
    // We keep track of the selection history here.
    private ArrayList<Integer> selectedImageHistory;
    // This is the index of the current selection in the history. If we are
    // going back and forth, we move this index and the user can easily browse
    // through the browsing history.
    private int selectedImageIndexInHistory = 0;
    // These maps map the paths to specific instance indexes.
    private HashMap<String, Integer> pathIndexMap = null;
    private HashMap<String, Integer> pathIndexMapThumbnail = null;
    // Lists of image paths and image thumbnail paths.
    private ArrayList<String> imgPaths = null;
    private ArrayList<String> imgThumbPaths = null;
    // The current neighborhood size.
    private volatile int neighborhoodSize = 5;
    // Statistics related to the kNN topology and the hubness of the data.
    // Percentage of points that occur at least once as neighbors, over all
    // neighborhood sizes.
    private float[] aboveZeroArray = null;
    // Neighbor occurrence distribution skewness, over all neighborhood sizes.
    private float[] skewArray = null;
    // Neighbor occurrence distribution kurtosis, over all neighborhood sizes.
    private float[] kurtosisArray = null;
    // Highest neighbor occurrence counts, over all neighborhood sizes.
    private float[][] highestHubnesses = null;
    // Indexes of top hubs in the data, over all neighborhood sizes.
    private int[][] highestHubIndexes = null;
    // kNN set entropies, over all neighborhood sizes.
    private float[] kEntropies = null;
    // Reverse kNN set entropies, over all neighborhood sizes.
    private float[] reverseKNNEntropies = null;
    // kNN set entropy skewness, over all neighborhood sizes.
    private float[] kEntropySkews = null;
    // Reverse kNN set entropy skews, over all neighborhood sizes.
    private float[] reverseKNNEntropySkews = null;
    // Label mismatch percentages in kNN sets, over all neighborhood sizes.
    private float[] badHubnessArray = null;
    // Global class to class hubness, over all neighborhood sizes.
    private float[][][] globalClassToClasshubness = null;
    // Percentages of points that are hubs, over all neighborhood sizes.
    private float[] hubPercs = null;
    // Percentages of points that are orphans, over all neighborhood sizes.
    private float[] orphanPercs = null;
    // Percentages of points that are regular points, over all neighborhood
    // sizes.
    private float[] regularPercs = null;
    // Occurrence distributions as histograms with a fixed bucket width.
    private int[][] bucketedOccurrenceDistributions = null;
    // The current query image.
    private BufferedImage queryImage;
    // The representation of the current query.
    private DataInstance queryImageRep;
    // Local image features of the current query, if available.
    private LFeatRepresentation queryImageLFeat;
    // Neighbors of the current query image.
    private int[] queryImageNeighbors;
    // Distances to the neighbors of the current query image.
    private float[] queryImageNeighborDists;
    // Neighborhood size used for the current query.
    private int kQuery = 10;
    // Whether the kNN stats have already been calculated or not.
    private boolean neighborStatsCalculated = false;
    // Whether the classifier models have already been trained or not.
    private boolean trainedModels = false;
    // A list of classifiers.
    private Classifier[] classifiers;
    // A list of classifier names for display, as these may differ from the
    // implementaiton names.
    private String[] classifierNameList = {"kNN", "FNN", "NWKNN", "AKNN",
        "hw-kNN", "h-FNN", "HIKNN", "NHBNN"};
    // Lists of top hub, good hub and bad hub indexes for each class for all
    // neighborhood sizes.
    private ArrayList<Integer>[][] classTopHubLists;
    private ArrayList<Integer>[][] classTopGoodHubsList;
    private ArrayList<Integer>[][] classTopBadHubsList;
    // Corresponding occurrence frequencies to the above lists.
    private ArrayList<Integer>[][] classHubnessArrValues;
    private ArrayList<Integer>[][] classHubnessArrGoodValues;
    private ArrayList<Integer>[][] classHubnessArrBadValues;
    // A list of image indexes that belongs to each particular class.
    private ArrayList<Integer>[] classImageIndexes = new ArrayList[numClasses];
    // Point type distributions for all classes.
    private float[][] classPTypes;
    // Per-class visualizations of the influence of hubness.
    private ClassHubsPanel[] classStatsOverviews;
    // Image coordinates for the MDS screen.
    private float[][] imageCoordinatesXY;
    // Number of images to show in the MDS screen.
    private int numImagesDrawn = 300;
    // Maximum and minimum display scale.
    private int maxImageScale = 80;
    private int minImageScale = 10;
    // Calculated MDS landscapes. Calculating a landscape takes some time, so
    // this helps with quickly changing the background when the neighborhood
    // size is changed, as previously calculated backgrounds are then shown
    // instead of being re-calculated.
    private BufferedImage[] mdsBackgrounds;
    // kNN graphs for all the k-values.
    private DirectedGraph<ImageNode, NeighborLink>[] neighborGraphs;
    private ArrayList<Integer> vertexIndexes;
    private ArrayList<ImageNode> vertices;
    private ArrayList<NeighborLink>[] edges;
    private HashMap<Integer, ImageNode> verticesHash;
    private HashMap<Integer, Integer> verticesNodeIndexHash;
    private VisualizationViewer[] graphVServers;
    // File containing the codebook profile for visual word utility assessment.
    File codebookProfileFile = null;
    // Panels for codebook vector profiles.
    CodebookVectorProfilePanel[] codebookProfPanels;

    /**
     * Nodes for kNN graph visualizations.
     */
    static class ImageNode {

        int id;
        ImageIcon icon;
        String thumbPath;

        /**
         * Initialization.
         *
         * @param id Integer that is the node ID and will correspond to the
         * index of the image in the representation.
         * @param icon ImageIcon that is the image thumbnail.
         * @param thumbPath String that is the thumbnail path.
         */
        public ImageNode(int id, ImageIcon icon, String thumbPath) {
            this.icon = icon;
            this.id = id;
            this.thumbPath = thumbPath;
        }

        /**
         * @return ImageIcon that is the image thumbnail.
         */
        public Icon getIcon() {
            return icon;
        }

        @Override
        public String toString() {
            return thumbPath;
        }
    }

    /**
     * Edges for kNN graph visualization.
     */
    static class NeighborLink {

        // The total number of edges.
        private static int edgeCount = 0;
        // Edge weight.
        double weight;
        // Edge ID.
        int id;
        // Source and target ImageNode that are connected due to the neighbor
        // relation.
        ImageNode source, target;

        /**
         * Initialization.
         *
         * @param weight Double that is the edge weight.
         * @param source ImageNode that is the source vertex for this edge.
         * @param target ImageNode that is the target vertex for this edge.
         */
        public NeighborLink(double weight, ImageNode source, ImageNode target) {
            this.id = edgeCount++;
            this.weight = weight;
            this.source = source;
            this.target = target;
        }

        @Override
        public String toString() {
            return " " + BasicMathUtil.makeADecimalCutOff(weight, 3);
        }

        /**
         * @return ImageNode that is the source vertex for this edge.
         */
        public ImageNode getSource() {
            return source;
        }

        /**
         * @return ImageNode that is the target vertex for this edge.
         */
        public ImageNode getTarget() {
            return target;
        }
    }

    /**
     * This method deletes and resets all the kNN graphs.
     */
    private void graphsDelete() {
        if (neighborGraphs != null) {
            for (int i = 0; i < neighborGraphs.length; i++) {
                neighborGraphs[i] = null;
                graphVServers[i] = null;
            }
        }
        neighborGraphs = null;
        vertexIndexes = null;
        vertices = null;
        edges = null;
        neighborGraphScrollPane.setViewportView(null);
        neighborGraphScrollPane.revalidate();
        neighborGraphScrollPane.repaint();
        System.gc();
    }

    /**
     * This method removes the currently selected image from the kNN graph
     * visualizations for all neighborhood sizes.
     */
    private void removeSelectedImageFromGraph() {
        if (neighborStatsCalculated && neighborGraphs != null
                && selectedImageHistory != null
                && selectedImageHistory.size() > 0) {
            int index = selectedImageHistory.get(selectedImageIndexInHistory);
            if (verticesHash.containsKey(index)) {
                ImageNode delVertex = verticesHash.get(index);
                // Create two lists of edges, those that need to be deleted and
                // those that will be retained. Create one such list for each
                // neighborhood size.
                ArrayList<NeighborLink>[] retainedEdges;
                ArrayList<NeighborLink>[] discardedEdges;
                retainedEdges = new ArrayList[50];
                discardedEdges = new ArrayList[50];
                for (int kTmp = 0; kTmp < 50; kTmp++) {
                    retainedEdges[kTmp] = new ArrayList<>(500);
                    discardedEdges[kTmp] = new ArrayList<>(500);
                    for (int i = 0; i < edges[kTmp].size(); i++) {
                        if (edges[kTmp].get(i) != null) {
                            if (delVertex.equals(edges[kTmp].get(i).getSource())
                                    || delVertex.equals(
                                    edges[kTmp].get(i).getTarget())) {
                                // Discard the edge.
                                discardedEdges[kTmp].add(edges[kTmp].get(i));
                            } else {
                                // Keep the edge.
                                retainedEdges[kTmp].add(edges[kTmp].get(i));
                            }
                        }
                    }
                    // Update the internal edge lists.
                    edges[kTmp] = retainedEdges[kTmp];
                    // Remove the removed edges from the graphs.
                    for (int i = 0; i < discardedEdges[kTmp].size(); i++) {
                        neighborGraphs[kTmp].removeEdge(
                                discardedEdges[kTmp].get(i));
                    }
                    neighborGraphs[kTmp].removeVertex(delVertex);
                    graphVServers[kTmp].revalidate();
                    graphVServers[kTmp].repaint();
                }
                verticesHash.remove(index);
            }
            // Update the graphical components.
            neighborGraphScrollPane.revalidate();
            neighborGraphScrollPane.repaint();
        }
    }

    /**
     * Add a list of images to the kNN graph visualizations.
     *
     * @param indexes ArrayList<Integer> of image indexes to insert.
     */
    private void addSelectedImagesToGraph(ArrayList<Integer> indexes) {
        int[] aIndexes = new int[0];
        if (indexes != null) {
            aIndexes = new int[indexes.size()];
            for (int i = 0; i < indexes.size(); i++) {
                aIndexes[i] = indexes.get(i);
            }
        }
        // Delegate to a method that operates on the index arrays.
        addSelectedImagesToGraph(aIndexes);
    }

    /**
     * Add an array of images to the kNN graph visualizations.
     *
     * @param indexes int[] of image indexes to insert.
     */
    private void addSelectedImagesToGraph(int[] indexes) {
        if (neighborStatsCalculated && neighborGraphs != null) {
            for (int index : indexes) {
                if (verticesHash.containsKey(index)) {
                    // If the image is already contained in the graphs, skip it.
                    continue;
                }
                // Create a new node to insert.
                ImageNode newVertex = new ImageNode(
                        index, new ImageIcon(imgThumbPaths.get(index)),
                        imgThumbPaths.get(index));
                // Add the node to the vertex set.
                vertexIndexes.add(index);
                vertices.add(newVertex);
                for (int kTmp = 0; kTmp < 50; kTmp++) {
                    neighborGraphs[kTmp].addVertex(newVertex);
                    verticesHash.put(index, newVertex);
                    verticesNodeIndexHash.put(index, vertices.size() - 1);
                    graphVServers[kTmp].revalidate();
                    graphVServers[kTmp].repaint();
                    neighborGraphScrollPane.revalidate();
                    neighborGraphScrollPane.repaint();
                }
            }
            // This might be improved. All edges are removed from the graphs
            // and then inserted anew.
            for (int kTmp = 0; kTmp < 50; kTmp++) {
                for (int i = 0; i < edges[kTmp].size(); i++) {
                    neighborGraphs[kTmp].removeEdge(edges[kTmp].get(i));
                }
                edges[kTmp] = new ArrayList<>(100);
                graphVServers[kTmp].revalidate();
                graphVServers[kTmp].repaint();
                neighborGraphScrollPane.revalidate();
                neighborGraphScrollPane.repaint();
            }
            NeighborSetFinder nsf = getNSF();
            int[][] kneighbors = nsf.getKNeighbors();
            float[][] kdistances = nsf.getKDistances();
            // For all the neighborhood sizes in the range.
            for (int kTmp = 0; kTmp < 50; kTmp++) {
                for (int i = 0; i < vertices.size(); i++) {
                    // For all the neighbors.
                    for (int kN = 0; kN < kTmp + 1; kN++) {
                        if (verticesHash.containsKey(
                                kneighbors[vertexIndexes.get(i)][kN])) {
                            NeighborLink newEdge =
                                    new NeighborLink(kdistances[
                                    vertexIndexes.get(i)][kN], vertices.get(i),
                                    verticesHash.get(kneighbors[
                                    vertexIndexes.get(i)][kN]));
                            neighborGraphs[kTmp].addEdge(newEdge,
                                    vertices.get(i), verticesHash.get(
                                    kneighbors[vertexIndexes.get(i)][kN]),
                                    EdgeType.DIRECTED);
                            edges[kTmp].add(newEdge);
                        }
                    }
                }
                graphVServers[kTmp].revalidate();
                graphVServers[kTmp].repaint();
            }
            // Determine how to display the nodes.
            Layout<ImageNode, NeighborLink> layout =
                    new CircleLayout(neighborGraphs[neighborhoodSize - 1]);
            layout.setSize(new Dimension(500, 500));
            layout.initialize();
            VisualizationViewer<ImageNode, NeighborLink> vv =
                    new VisualizationViewer<>(layout);
            vv.setPreferredSize(new Dimension(550, 550));
            vv.setMinimumSize(new Dimension(550, 550));
            vv.setDoubleBuffered(true);
            vv.setEnabled(true);
            graphVServers[neighborhoodSize - 1] = vv;
            vv.getRenderContext().setVertexIconTransformer(
                    new IconTransformer<ImageNode, Icon>());
            vv.getRenderContext().setVertexShapeTransformer(
                    new ShapeTransformer<ImageNode, Shape>());
            vv.getRenderContext().setEdgeArrowPredicate(
                    new DirectionDisplayPredicate());
            vv.getRenderContext().setEdgeLabelTransformer(
                    new Transformer() {
                @Override
                public String transform(Object e) {
                    return (e.toString());
                }
            });
            PluggableGraphMouse gm = new PluggableGraphMouse();
            gm.add(new PickingGraphMousePlugin());
            vv.setGraphMouse(gm);
            vv.setBackground(Color.WHITE);
            vv.setVisible(true);
            final PickedState<ImageNode> pickedState =
                    vv.getPickedVertexState();
            pickedState.addItemListener(new ItemListener() {
                @Override
                public void itemStateChanged(ItemEvent e) {
                    Object subject = e.getItem();
                    if (subject instanceof ImageNode) {
                        ImageNode vertex = (ImageNode) subject;
                        if (pickedState.isPicked(vertex)) {
                            setSelectedImageForIndex(vertex.id);
                        } else {
                        }
                    }
                }
            });
            vv.validate();
            vv.repaint();
            neighborGraphScrollPane.setViewportView(vv);
            neighborGraphScrollPane.revalidate();
            neighborGraphScrollPane.repaint();
        }
    }

    /**
     * Inserts the currently selected image into all kNN graphs for
     * visualization.
     */
    private void addSelectedImageToGraph() {
        if (neighborStatsCalculated && neighborGraphs != null
                && selectedImageHistory != null
                && selectedImageHistory.size() > 0) {
            int index = selectedImageHistory.get(selectedImageIndexInHistory);
            if (verticesHash.containsKey(index)) {
                // If it is already contained within the graphs, do nothing.
                return;
            }
            // Create a new node for the image.
            ImageNode newVertex = new ImageNode(index,
                    new ImageIcon(imgThumbPaths.get(index)),
                    imgThumbPaths.get(index));
            vertexIndexes.add(index);
            vertices.add(newVertex);
            NeighborSetFinder nsf = getNSF();
            int[][] kneighbors = nsf.getKNeighbors();
            float[][] kdistances = nsf.getKDistances();
            // For all relevant neighborhood sizes.
            for (int kTmp = 0; kTmp < 50; kTmp++) {
                neighborGraphs[kTmp].addVertex(newVertex);
                verticesHash.put(index, newVertex);
                verticesNodeIndexHash.put(index, vertices.size() - 1);
                // Re-draw all the edges.
                // This should be changed to only update the affected ones.
                for (int i = 0; i < edges[kTmp].size(); i++) {
                    neighborGraphs[kTmp].removeEdge(edges[kTmp].get(i));
                }
                for (int i = 0; i < vertices.size(); i++) {
                    for (int kN = 0; kN < kTmp; kN++) {
                        if (verticesHash.containsKey(
                                kneighbors[vertexIndexes.get(i)][kN])) {
                            NeighborLink newEdge =
                                    new NeighborLink(
                                    kdistances[vertexIndexes.get(i)][kN],
                                    vertices.get(i),
                                    verticesHash.get(
                                    kneighbors[vertexIndexes.get(i)][kN]));
                            neighborGraphs[kTmp].addEdge(newEdge,
                                    vertices.get(i),
                                    verticesHash.get(
                                    kneighbors[vertexIndexes.get(i)][kN]),
                                    EdgeType.DIRECTED);
                            edges[kTmp].add(newEdge);
                        }
                    }
                }
                graphVServers[kTmp].revalidate();
                graphVServers[kTmp].repaint();
            }
            // Set up how the nodes will be drawn.
            Layout<ImageNode, NeighborLink> layout = new CircleLayout(
                    neighborGraphs[neighborhoodSize - 1]);
            layout.setSize(new Dimension(500, 500));
            layout.initialize();
            VisualizationViewer<ImageNode, NeighborLink> vv =
                    new VisualizationViewer<>(layout);
            vv.setPreferredSize(new Dimension(550, 550));
            vv.setMinimumSize(new Dimension(550, 550));
            vv.setDoubleBuffered(true);
            vv.setEnabled(true);
            graphVServers[neighborhoodSize - 1] = vv;
            vv.getRenderContext().setVertexIconTransformer(
                    new IconTransformer<ImageNode, Icon>());
            vv.getRenderContext().setVertexShapeTransformer(
                    new ShapeTransformer<ImageNode, Shape>());
            vv.getRenderContext().setEdgeArrowPredicate(
                    new DirectionDisplayPredicate());
            vv.getRenderContext().setEdgeLabelTransformer(
                    new Transformer() {
                @Override
                public String transform(Object e) {
                    return (e.toString());
                }
            });
            PluggableGraphMouse gm = new PluggableGraphMouse();
            gm.add(new PickingGraphMousePlugin());
            vv.setGraphMouse(gm);
            vv.setBackground(Color.WHITE);
            vv.setVisible(true);
            final PickedState<ImageNode> pickedState =
                    vv.getPickedVertexState();
            pickedState.addItemListener(new ItemListener() {
                @Override
                public void itemStateChanged(ItemEvent e) {
                    Object subject = e.getItem();
                    if (subject instanceof ImageNode) {
                        ImageNode vertex = (ImageNode) subject;
                        if (pickedState.isPicked(vertex)) {
                            setSelectedImageForIndex(vertex.id);
                        }
                    }
                }
            });
            // Refresh all the display components.
            vv.revalidate();
            vv.repaint();
            neighborGraphScrollPane.setViewportView(
                    graphVServers[neighborhoodSize - 1]);
            neighborGraphScrollPane.revalidate();
            neighborGraphScrollPane.repaint();
        }
    }

    /**
     * IconTransformer class. These class is used for node visualization in kNN
     * graphs.
     *
     * @param <ImageNode> Image node.
     * @param <Icon> Corresponding icon.
     */
    class IconTransformer<ImageNode, Icon>
            implements Transformer<ImageNode, Icon> {

        public static final int ICON_SIZE = 30;

        /**
         * @return Integer that is the shape height.
         */
        public int getHeight() {
            return (ICON_SIZE);
        }

        /**
         * @return Integer that is the shape width.
         */
        public int getWidth() {
            return (ICON_SIZE);
        }

        @Override
        public Icon transform(ImageNode node) {
            if (node != null) {
                Icon icon = (Icon) (new ImageIcon(node.toString()));
                return icon;
            } else {
                return null;
            }
        }
    }

    /**
     * ShapeTransformer class. These class is used for node visualization in kNN
     * graphs.
     *
     * @param <ImageNode> Image node.
     * @param <Shape> Corresponding Shape.
     */
    class ShapeTransformer<ImageNode, Shape>
            implements Transformer<ImageNode, Shape> {

        public static final int ICON_SIZE = 30;

        /**
         * @return Integer that is the shape height.
         */
        public int getHeight() {
            return (ICON_SIZE);
        }

        /**
         * @return Integer that is the shape width.
         */
        public int getWidth() {
            return (ICON_SIZE);
        }

        @Override
        public Shape transform(ImageNode node) {
            if (node != null) {
                ImageIcon icon = new ImageIcon(node.toString());
                int width = icon.getIconWidth();
                int height = icon.getIconHeight();
                Rectangle2D shape = new Rectangle2D.Float(
                        -width / 2, -height / 2, width, height);
                return (Shape) shape;
            } else {
                return null;
            }
        }
    }

    /**
     * Handler for directed and undirected edges for kNN graph visualization.
     *
     * @param <V> Vertex type.
     * @param <E> Edge type.
     */
    private final static class DirectionDisplayPredicate<V, E>
            implements Predicate<Context<Graph<V, E>, E>> {

        /**
         * The default constructor.
         */
        public DirectionDisplayPredicate() {
        }

        @Override
        public boolean evaluate(Context<Graph<V, E>, E> context) {
            Graph<V, E> graph = context.graph;
            E edge = context.element;
            if (graph.getEdgeType(edge) == EdgeType.DIRECTED) {
                return true;
            }
            if (graph.getEdgeType(edge) == EdgeType.UNDIRECTED) {
                return true;
            }
            return true;
        }
    }

    /**
     * Initialize all the kNN graphs for visualization.
     */
    private void graphsInit() {
        neighborGraphs = new DirectedGraph[50];
        graphVServers = new VisualizationViewer[50];
        edges = new ArrayList[50];
        // For all the relevant neighborhood sizes.
        for (int kTmp = 0; kTmp < 50; kTmp++) {
            // Create a new graph.
            DirectedGraph graph = new DirectedSparseMultigraph<>();
            neighborGraphs[kTmp] = graph;
            Layout<ImageNode, NeighborLink> layout = new CircleLayout(
                    neighborGraphs[kTmp]);
            layout.setSize(new Dimension(500, 500));
            // Set the rendering specification.
            VisualizationViewer<ImageNode, NeighborLink> vv =
                    new VisualizationViewer<>(layout);
            vv.setPreferredSize(new Dimension(550, 550));
            vv.setMinimumSize(new Dimension(550, 550));
            vv.setDoubleBuffered(true);
            vv.setEnabled(true);
            graphVServers[kTmp] = vv;
            vv.getRenderContext().setVertexIconTransformer(
                    new IconTransformer<ImageNode, Icon>());
            vv.getRenderContext().setVertexShapeTransformer(
                    new ShapeTransformer<ImageNode, Shape>());
            vv.getRenderContext().setEdgeArrowPredicate(
                    new DirectionDisplayPredicate());
            vv.getRenderContext().setEdgeLabelTransformer(new Transformer() {
                @Override
                public String transform(Object e) {
                    return (e.toString());
                }
            });
            PluggableGraphMouse gm = new PluggableGraphMouse();
            gm.add(new PickingGraphMousePlugin());
            vv.setGraphMouse(gm);
            vv.setBackground(Color.WHITE);
            vv.setVisible(true);
            final PickedState<ImageNode> pickedState =
                    vv.getPickedVertexState();
            // Add the selection listeners.
            pickedState.addItemListener(new ItemListener() {
                @Override
                public void itemStateChanged(ItemEvent e) {
                    Object subject = e.getItem();
                    if (subject instanceof ImageNode) {
                        ImageNode vertex = (ImageNode) subject;
                        if (pickedState.isPicked(vertex)) {
                            setSelectedImageForIndex(vertex.id);
                        }
                    }
                }
            });
        }
        verticesHash = new HashMap<>(500);
        verticesNodeIndexHash = new HashMap<>(500);
        vertexIndexes = new ArrayList<>(200);
        vertices = new ArrayList<>(200);
        edges = new ArrayList[50];
        for (int kTmp = 0; kTmp < 50; kTmp++) {
            edges[kTmp] = new ArrayList<>(500);
        }
        // Refresh the display.
        graphVServers[neighborhoodSize - 1].revalidate();
        graphVServers[neighborhoodSize - 1].repaint();
        neighborGraphScrollPane.setViewportView(
                graphVServers[neighborhoodSize - 1]);
        neighborGraphScrollPane.setVisible(true);
        neighborGraphScrollPane.revalidate();
        neighborGraphScrollPane.repaint();
    }

    /**
     * Train all the classifier models.
     */
    public void trainModels() {
        if (busyCalculating) {
            // If the system is already working on something, this call will be
            // ignored.
            return;
        }
        busyCalculating = true;
        try {
            trainedModels = true;
            classifiers = new Classifier[classifierNameList.length];
            // Get the current metric context.
            CombinedMetric cmet = getCombinedMetric();
            // Initialize all the classifiers.
            classifiers[0] = new KNN(kQuery, cmet);
            classifiers[1] = new FNN(kQuery, cmet, numClasses);
            classifiers[2] = new NWKNN(kQuery, cmet, numClasses);
            classifiers[3] = new AKNN(kQuery, cmet, numClasses);
            classifiers[4] = new HwKNN(numClasses, cmet, kQuery);
            classifiers[5] = new DWHFNN(kQuery, cmet,
                    numClasses);
            classifiers[6] = new HIKNN(kQuery, cmet, numClasses);
            classifiers[7] = new NHBNN(kQuery, cmet, numClasses);
            // Get the current distance matrix.
            float[][] distances = getDistances();
            // Get the current kNN sets.
            NeighborSetFinder nsf = getNSF();
            System.out.println("Training classifier models.");
            ArrayList<Integer> completeDataArray = new ArrayList<>(
                    quantizedRepresentation.size());
            for (int i = 0; i < quantizedRepresentation.size(); i++) {
                completeDataArray.add(i);
            }
            for (int i = 0; i < classifiers.length; i++) {
                // Set the data.
                classifiers[i].setDataIndexes(completeDataArray,
                        quantizedRepresentation);
                if (classifiers[i] instanceof DistMatrixUserInterface) {
                    // If the classifier requires the distance matrix, set the
                    // distance matrix.
                    ((DistMatrixUserInterface) (classifiers[i])).
                            setDistMatrix(distances);
                }
                if (classifiers[i] instanceof NSFUserInterface) {
                    // If the classifier requires kNN sets, set the kNN sets.
                    ((NSFUserInterface) (classifiers[i])).setNSF(nsf);
                }
                try {
                    classifiers[i].train();
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
            System.out.println("Models trained.");
            JOptionPane.showMessageDialog(frameReference, "Models trained.");
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            // Report that the calculations have been finished.
            busyCalculating = false;
        }
    }

    /**
     * Sets the currently selected image to be the query image in the query
     * panel.
     */
    public void setQueryImageFromCollection() {
        if (busyCalculating) {
            // If the system is already working on something, this call will be
            // ignored.
            return;
        }
        if (selectedImageHistory == null
                || selectedImageIndexInHistory >= selectedImageHistory.size()) {
            return;
        }
        try {
            int index = selectedImageHistory.get(selectedImageIndexInHistory);
            queryImage = getPhoto(index);
            // Set the image to the query image panel.
            queryImagePanel.setImage(queryImage);
            queryImagePanel.revalidate();
            queryImagePanel.repaint();
            if (quantizedRepresentation != null) {
                // Use the existing representation, if available.
                queryImageRep = quantizedRepresentation.getInstance(index);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * This method queries the image dataset by a single specified image.
     */
    private void imageQuery() {
        if (busyCalculating) {
            // If the system is already working on something, this call will be
            // ignored.
            return;
        }
        try {
            busyCalculating = true;
            // Get the k-nearest neighbors.
            NeighborSetFinder nsf = getNSF();
            queryImageNeighbors = NeighborSetFinder.getIndexesOfNeighbors(
                    quantizedRepresentation, queryImageRep,
                    Math.min(kQuery, nsf.getKNeighbors()[0].length),
                    getCombinedMetric());
            CombinedMetric cmet = getCombinedMetric();
            queryImageNeighborDists = new float[queryImageNeighbors.length];
            // Get the distances to the neighbors.
            for (int i = 0; i < queryImageNeighbors.length; i++) {
                queryImageNeighborDists[i] = cmet.dist(queryImageRep,
                        quantizedRepresentation.getInstance(
                        queryImageNeighbors[i]));
            }
            queryNNPanel.removeAll();
            queryNNPanel.revalidate();
            queryNNPanel.repaint();
            // Add all the kNN-s of the query to the query panel.
            for (int i = 0; i < queryImageNeighbors.length; i++) {
                BufferedImage thumb = thumbnails.get(queryImageNeighbors[i]);
                ImagePanelWithClass imgPan =
                        new ImagePanelWithClass(classColors);
                imgPan.addMouseListener(new NeighborSelectionListener());
                imgPan.setImage(thumb,
                        quantizedRepresentation.getLabelOf(
                        queryImageNeighbors[i]),
                        queryImageNeighbors[i]);
                queryNNPanel.add(imgPan);
            }
            queryNNPanel.revalidate();
            queryNNPanel.repaint();
            // If the classifier models are available, get some predictions and
            // display them to the user.
            if (trainedModels) {
                System.out.println("Classifying.");
                classifierPredictionsPanel.removeAll();
                classifierPredictionsPanel.revalidate();
                classifierPredictionsPanel.repaint();
                // Calculate the distances to the remaining points.
                float[] trainingDists =
                        new float[quantizedRepresentation.size()];
                for (int i = 0; i < queryImageNeighbors.length; i++) {
                    trainingDists[queryImageNeighbors[i]] =
                            queryImageNeighborDists[i];
                }
                for (int i = 0; i < classifiers.length; i++) {
                    System.out.println("Classification by"
                            + classifierNameList[i]);
                    // Class affiliation prediction.
                    float[] prediction;
                    if (classifiers[i] instanceof
                            NeighborPointsQueryUserInterface) {
                        // Get the prediction
                        prediction =
                                ((NeighborPointsQueryUserInterface) (
                                classifiers[i])).classifyProbabilistically(
                                queryImageRep,
                                trainingDists,
                                queryImageNeighbors);
                    } else {
                        // Get the prediction.
                        prediction = classifiers[i].classifyProbabilistically(
                                queryImageRep);
                    }
                    ClassifierResultPanel cResPanel =
                            new ClassifierResultPanel();
                    cResPanel.setResults(prediction,
                            classifierNameList[i], classColors, classNames);
                    classifierPredictionsPanel.add(cResPanel);
                }
                classifierPredictionsPanel.revalidate();
                classifierPredictionsPanel.repaint();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            busyCalculating = false;
        }
    }

    /**
     * Sets all the neighbor stats for the currently selected k-value.
     *
     * @param currentK Integer that is the current neighborhood size.
     */
    public synchronized void setStatFieldsForK(int currentK) {
        // Percentage of elements that occur at least ones.
        if (aboveZeroArray != null) {
            percAboveLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    aboveZeroArray[currentK - 1], 2))).toString());
        } else {
            percAboveLabelValue.setText("...");
        }
        // Neighbor occurrence distribution skewness.
        if (skewArray != null) {
            skewnessLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    skewArray[currentK - 1], 2))).toString());
        } else {
            skewnessLabelValue.setText("...");
        }
        // Neighbor occurrence distribution kurtosis.
        if (kurtosisArray != null) {
            kurtosisLabelValue.setText(
                    (new Float(BasicMathUtil.makeADecimalCutOff(
                    kurtosisArray[currentK - 1], 2))).toString());
        } else {
            kurtosisLabelValue.setText("...");
        }
        // Highest neighbor occurrence frequencies.
        if (highestHubnesses != null) {
            majorDegLabelValue.setText(
                    (new Float(BasicMathUtil.makeADecimalCutOff(
                    highestHubnesses[currentK - 1][
                    highestHubnesses[currentK - 1].length - 1], 2))).
                    toString());
        } else {
            majorDegLabelValue.setText("...");
        }
        // kNN set entropies.
        if (kEntropies != null) {
            nkEntropySkewnessValues.setText(
                    (new Float(BasicMathUtil.makeADecimalCutOff(
                    kEntropySkews[currentK - 1], 2))).toString());
            nkEntropyLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    kEntropies[currentK - 1], 2))).toString());
        } else {
            nkEntropyLabelValue.setText("...");
            nkEntropySkewnessValues.setText("...");
        }
        // Reverse kNN set entropies.
        if (reverseKNNEntropies != null) {
            rnkEntropySkewnessValue.setText(
                    (new Float(BasicMathUtil.makeADecimalCutOff(
                    reverseKNNEntropySkews[currentK - 1], 2))).toString());
            rnkEntropyValue.setText((new Float(BasicMathUtil.makeADecimalCutOff(
                    reverseKNNEntropies[currentK - 1], 2))).toString());
        } else {
            rnkEntropyValue.setText("...");
            rnkEntropySkewnessValue.setText("...");
        }
        // Bad hubness percentages as percentages of label mismatches in kNN
        // sets.
        if (badHubnessArray != null) {
            badHubnessLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    badHubnessArray[currentK - 1], 2))).toString());
        } else {
            badHubnessLabelValue.setText("...");
        }
        // Percentage of points that are hubs.
        if (hubPercs != null) {
            hubsLabelValue.setText((new Float(BasicMathUtil.makeADecimalCutOff(
                    hubPercs[currentK - 1], 2))).toString());
        } else {
            hubsLabelValue.setText("...");
        }
        // Percentage of points that are orphans.
        if (orphanPercs != null) {
            orphansLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    orphanPercs[currentK - 1], 2))).toString());
        } else {
            orphansLabelValue.setText("...");
        }
        // Percentage of points that are regular points.
        if (regularPercs != null) {
            regularLabelValue.setText((new Float(BasicMathUtil.
                    makeADecimalCutOff(
                    regularPercs[currentK - 1], 2))).toString());
        } else {
            regularLabelValue.setText("...");
        }
        // Refresh the display.
        percAboveLabelValue.revalidate();
        percAboveLabelValue.repaint();
        skewnessLabelValue.revalidate();
        skewnessLabelValue.repaint();
        kurtosisLabelValue.revalidate();
        kurtosisLabelValue.repaint();
        majorDegLabelValue.revalidate();
        majorDegLabelValue.repaint();
        nkEntropySkewnessValues.revalidate();
        nkEntropySkewnessValues.repaint();
        nkEntropyLabelValue.revalidate();
        nkEntropyLabelValue.repaint();
        rnkEntropySkewnessValue.revalidate();
        rnkEntropySkewnessValue.repaint();
        rnkEntropyValue.revalidate();
        rnkEntropyValue.repaint();
        badHubnessLabelValue.revalidate();
        badHubnessLabelValue.repaint();
        hubsLabelValue.revalidate();
        hubsLabelValue.repaint();
        orphansLabelValue.revalidate();
        orphansLabelValue.repaint();
        regularLabelValue.revalidate();
        regularLabelValue.repaint();
        // Now generate the occurrence frequency distribution chart, discretized
        // to fixed-length buckets.
        DefaultCategoryDataset hDistDataset = new DefaultCategoryDataset();
        for (int i = 0; i < bucketedOccurrenceDistributions[
                    neighborhoodSize - 1].length; i++) {
            hDistDataset.addValue(
                    bucketedOccurrenceDistributions[
                         neighborhoodSize - 1][i], "Number of Examples",
                    i + "");
        }
        JFreeChart chart = ChartFactory.createBarChart(
                "Occurrence Frequency Distribution", "", "",
                hDistDataset, PlotOrientation.VERTICAL, false, true, false);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(440, 180));
        chartHoldingPanelOccDistribution.removeAll();
        chartHoldingPanelOccDistribution.add(chartPanel);
        chartHoldingPanelOccDistribution.revalidate();
        chartHoldingPanelOccDistribution.repaint();
        // Calculate class to class hubness.
        for (int c1 = 0; c1 < numClasses; c1++) {
            for (int c2 = 0; c2 < numClasses; c2++) {
                classHubnessTable.setValueAt(
                        globalClassToClasshubness[currentK - 1][c1][c2], c1,
                        c2);
            }
        }
        classHubnessTable.setDefaultRenderer(Object.class,
                new ClassToClassHubnessMatrixRenderer(
                globalClassToClasshubness[currentK - 1],
                numClasses));
    }

    /**
     * Neighbor selection listener.
     */
    class NeighborSelectionListener implements MouseListener {

        @Override
        public void mousePressed(MouseEvent e) {
            Component comp = e.getComponent();
            if (comp instanceof ImagePanelWithClass) {
                int index = ((ImagePanelWithClass) comp).getImageIndex();
                setSelectedImageForIndex(index);
            }
        }

        @Override
        public void mouseReleased(MouseEvent e) {
        }

        @Override
        public void mouseEntered(MouseEvent e) {
        }

        @Override
        public void mouseExited(MouseEvent e) {
        }

        @Override
        public void mouseClicked(MouseEvent e) {
            Component comp = e.getComponent();
            if (comp instanceof ImagePanelWithClass) {
                int index = ((ImagePanelWithClass) comp).getImageIndex();
                setSelectedImageForIndex(index);
            }
        }
    }

    /**
     * This class handles adding neighbors and reverse neighbors to their
     * panels.
     */
    private class SetImageNeighborsHelper implements Runnable {

        private BufferedImage thumb;
        private int label, index;
        private JPanel panel;

        /**
         * Initialization.
         *
         * @param panel Panel to add the image to.
         * @param thumb Thumbnail of the image to add.
         * @param label Label of the image to add.
         * @param index Integer that is the index of the image to add.
         */
        public SetImageNeighborsHelper(JPanel panel, BufferedImage thumb,
                int label, int index) {
            this.thumb = thumb;
            this.label = label;
            this.index = index;
            this.panel = panel;
        }

        @Override
        public void run() {
            ImagePanelWithClass rrneighbor =
                    new ImagePanelWithClass(classColors);
            rrneighbor.addMouseListener(new NeighborSelectionListener());
            rrneighbor.setImage(thumb, label, index);
            panel.add(rrneighbor);
        }
    }

    /**
     * This method sets the image of the specified index as the currently
     * selected image and updates all the views.
     *
     * @param index Integer that is the index of the image to select as the
     * current image.
     */
    private synchronized void setSelectedImageForIndex(int index) {
        try {
            // Update the selected image panels.
            BufferedImage photo = getPhoto(index);
            selectedImagePanelClassNeighborMain.setImage(photo);
            selectedImagePanelClassNeighbor.setImage(photo);
            selectedImagePanelClass.setImage(photo);
            selectedImagePanelSearch.setImage(photo);
            String shortPath = imgPaths.get(index).substring(
                    workspace.getPath().length(), imgPaths.get(index).length());
            // Update the labels with the new name.
            selectedImagePathLabelClassNeighborMain.setText(shortPath);
            selectedImagePathLabelClassNeighbor.setText(shortPath);
            selectedImagePathLabelClass.setText(shortPath);
            selectedImagePathLabelSearch.setText(shortPath);
            // Update the class colors.
            selectedImageLabelClassNeighborMain.setBackground(
                    classColors[quantizedRepresentation.getLabelOf(index)]);
            selectedImageLabelClassNeighbor.setBackground(
                    classColors[quantizedRepresentation.getLabelOf(index)]);
            selectedImageLabelClass.setBackground(
                    classColors[quantizedRepresentation.getLabelOf(index)]);
            selectedImageLabelSearch.setBackground(
                    classColors[quantizedRepresentation.getLabelOf(index)]);
            // Refresh the display.
            selectedImageLabelClassNeighborMain.setOpaque(true);
            selectedImageLabelClassNeighbor.setOpaque(true);
            selectedImageLabelClass.setOpaque(true);
            selectedImageLabelSearch.setOpaque(true);
            selectedImageLabelClassNeighborMain.repaint();
            selectedImageLabelClassNeighbor.repaint();
            selectedImageLabelClass.repaint();
            selectedImageLabelSearch.repaint();
            // Update the history.
            if (selectedImageHistory == null) {
                selectedImageHistory = new ArrayList<>(200);
                selectedImageIndexInHistory = -1;
            }
            // Discard the future history.
            if (selectedImageIndexInHistory < selectedImageHistory.size() - 1) {
                for (int i = selectedImageHistory.size() - 1;
                        i > selectedImageIndexInHistory; i--) {
                    selectedImageHistory.remove(i);
                }
            }
            selectedImageHistory.add(index);
            selectedImageIndexInHistory = selectedImageHistory.size() - 1;
            // Update the nearest neighbors and the reverse nearest neighbors.
            NeighborSetFinder nsf = getNSF();
            nnPanel.removeAll();
            rnnPanel.removeAll();
            nnPanel.revalidate();
            nnPanel.repaint();
            rnnPanel.revalidate();
            rnnPanel.repaint();
            int[][] kneighbors = nsf.getKNeighbors();
            for (int neighborIndex = 0; neighborIndex < neighborhoodSize;
                    neighborIndex++) {
                // Insert all the nearest neighbors to their panel.
                BufferedImage thumb =
                        thumbnails.get(kneighbors[index][neighborIndex]);
                try {
                    Thread t = new Thread(
                            new SetImageNeighborsHelper(
                            nnPanel, thumb,
                            quantizedRepresentation.getLabelOf(
                            kneighbors[index][neighborIndex]),
                            kneighbors[index][neighborIndex]));
                    t.start();
                    t.join(500);
                    if (t.isAlive()) {
                        t.interrupt();
                    }
                } catch (Throwable thr) {
                    System.err.println(thr.getMessage());
                }
            }
            // Insert all the reverse nearest neighbors to their panel.
            ArrayList<Integer>[] rrns = null;
            if (rnnSetsAllK != null) {
                rrns = rnnSetsAllK[neighborhoodSize - 1];
            }
            if (rrns != null && rrns[index] != null && rrns[index].size() > 0) {
                for (int i = 0; i < rrns[index].size(); i++) {
                    BufferedImage thumb = thumbnails.get(rrns[index].get(i));
                    try {
                        Thread t = new Thread(
                                new SetImageNeighborsHelper(
                                rnnPanel, thumb,
                                quantizedRepresentation.getLabelOf(
                                rrns[index].get(i)), rrns[index].get(i)));
                        t.start();
                        t.join(500);
                        if (t.isAlive()) {
                            t.interrupt();
                        }
                    } catch (Throwable thr) {
                        System.err.println(thr.getMessage());
                    }
                }
            }
            // Refresh the neighbor and reverse neighbor panels.
            nnPanel.revalidate();
            nnPanel.repaint();
            rnnPanel.revalidate();
            rnnPanel.repaint();
            // Visualize the neighbor occurrence profile of the selected image.
            DefaultPieDataset pieData = new DefaultPieDataset();
            for (int c = 0; c < numClasses; c++) {
                pieData.setValue(classNames[c],
                        occurrenceProfilesAllK[neighborhoodSize - 1][index][c]);
            }
            JFreeChart chart = ChartFactory.createPieChart3D("occurrence "
                    + "profile", pieData, true, true, false);
            PiePlot3D plot = (PiePlot3D) chart.getPlot();
            plot.setStartAngle(290);
            plot.setDirection(Rotation.CLOCKWISE);
            plot.setForegroundAlpha(0.5f);
            PieRenderer prend = new PieRenderer(classColors);
            prend.setColor(plot, pieData);
            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(240, 200));
            occProfileChartHolder.removeAll();
            occProfileChartHolder.add(chartPanel);
            occProfileChartHolder.revalidate();
            occProfileChartHolder.repaint();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * This method sets the image of the specified history index as the
     * currently selected image and updates all the views.
     *
     * @param index Integer that is the history index of the image to select as
     * the current image.
     */
    private synchronized void setSelectedImageForHistoryIndex(
            int historyIndex) {
        // Update the selected image panels.
        BufferedImage photo = getPhoto(selectedImageHistory.get(historyIndex));
        selectedImagePanelClassNeighborMain.setImage(photo);
        selectedImagePanelClassNeighbor.setImage(photo);
        selectedImagePanelClass.setImage(photo);
        selectedImagePanelSearch.setImage(photo);
        int index = selectedImageHistory.get(historyIndex);
        // Update the labels with the new name.
        String shortPath = imgPaths.get(index).substring(
                workspace.getPath().length(), imgPaths.get(index).length());
        selectedImagePathLabelClassNeighborMain.setText(shortPath);
        selectedImagePathLabelClassNeighbor.setText(shortPath);
        selectedImagePathLabelClass.setText(shortPath);
        selectedImagePathLabelSearch.setText(shortPath);
        // Update the class colors.
        selectedImageLabelClassNeighborMain.setBackground(
                classColors[quantizedRepresentation.getLabelOf(index)]);
        selectedImageLabelClassNeighbor.setBackground(
                classColors[quantizedRepresentation.getLabelOf(index)]);
        selectedImageLabelClass.setBackground(
                classColors[quantizedRepresentation.getLabelOf(index)]);
        selectedImageLabelSearch.setBackground(
                classColors[quantizedRepresentation.getLabelOf(index)]);
        // Refresh the display.
        selectedImageLabelClassNeighborMain.setOpaque(true);
        selectedImageLabelClassNeighbor.setOpaque(true);
        selectedImageLabelClass.setOpaque(true);
        selectedImageLabelSearch.setOpaque(true);
        selectedImageLabelClassNeighborMain.repaint();
        selectedImageLabelClassNeighbor.repaint();
        selectedImageLabelClass.repaint();
        selectedImageLabelSearch.repaint();
        // Update the nearest neighbors and the reverse nearest neighbors.
        NeighborSetFinder nsf = getNSF();
        nnPanel.removeAll();
        rnnPanel.removeAll();
        nnPanel.revalidate();
        nnPanel.repaint();
        rnnPanel.revalidate();
        rnnPanel.repaint();
        int[][] kneighbors = nsf.getKNeighbors();
        for (int neighborIndex = 0; neighborIndex < neighborhoodSize;
                neighborIndex++) {
            BufferedImage thumb = thumbnails.get(
                    kneighbors[index][neighborIndex]);
            try {
                Thread t = new Thread(new SetImageNeighborsHelper(
                        nnPanel, thumb, quantizedRepresentation.getLabelOf(
                        kneighbors[index][neighborIndex]),
                        kneighbors[index][neighborIndex]));
                t.start();
                t.join(500);
                if (t.isAlive()) {
                    t.interrupt();
                }
            } catch (Throwable thr) {
                System.err.println(thr.getMessage());
            }
        }
        ArrayList<Integer>[] rrns = rnnSetsAllK[neighborhoodSize - 1];
        if (rrns[index] != null && rrns[index].size() > 0) {
            for (int i = 0; i < rrns[index].size(); i++) {
                BufferedImage thumb = thumbnails.get(rrns[index].get(i));
                try {
                    Thread t = new Thread(
                            new SetImageNeighborsHelper(
                            rnnPanel, thumb,
                            quantizedRepresentation.getLabelOf(
                            rrns[index].get(i)), rrns[index].get(i)));
                    t.start();
                    t.join(500);
                    if (t.isAlive()) {
                        t.interrupt();
                    }
                } catch (Throwable thr) {
                    System.err.println(thr.getMessage());
                }
            }
        }
        // Refresh the neighbor and reverse neighbor panels.
        nnPanel.revalidate();
        nnPanel.repaint();
        rnnPanel.revalidate();
        rnnPanel.repaint();
        // Visualize the neighbor occurrence profile of the selected image.
        DefaultPieDataset pieData = new DefaultPieDataset();
        for (int c = 0; c < numClasses; c++) {
            pieData.setValue(classNames[c],
                    occurrenceProfilesAllK[neighborhoodSize - 1][index][c]);
        }
        JFreeChart chart = ChartFactory.createPieChart3D("occurrence profile",
                pieData, true, true, false);
        PiePlot3D plot = (PiePlot3D) chart.getPlot();
        plot.setStartAngle(290);
        plot.setDirection(Rotation.CLOCKWISE);
        plot.setForegroundAlpha(0.5f);
        PieRenderer prend = new PieRenderer(classColors);
        prend.setColor(plot, pieData);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(240, 200));
        occProfileChartHolder.removeAll();
        occProfileChartHolder.add(chartPanel);
        occProfileChartHolder.revalidate();
        occProfileChartHolder.repaint();
    }

    /**
     * This method gets the photo for the provided image index.
     *
     * @param index Integer that is the image index in the data.
     * @return BufferedImage corresponding to the image index.
     */
    private BufferedImage getPhoto(int index) {
        if (images[index] == null) {
            int pathFeatureIndex =
                    quantizedRepresentation.getIndexForAttributeName(
                    "relative_path");
            File inImageFile = new File(workspace, "photos"
                    + quantizedRepresentation.getInstance(index).sAttr[
                    pathFeatureIndex]);
            try {
                images[index] = ImageIO.read(inImageFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
        return images[index];
    }

    /**
     * This method returns the distances that are currently in use. If there is
     * a secondary distance matrix, it returns that one. If there is no
     * secondary distance matrix, it returns the primary matrix instead.
     *
     * @return float[][] that is the currently used distance matrix.
     */
    private float[][] getDistances() {
        if (distMatrixSecondary != null) {
            return distMatrixSecondary;
        } else {
            return distMatrixPrimary;
        }
    }

    /**
     * This method returns the currently employed CombinedMetric object for
     * distance calculations. If there are secondary distances in use, it
     * returns the secondary CombinedMetric object. If not, the primary one.
     *
     * @return CombinedMetric object that is currently in use.
     */
    private CombinedMetric getCombinedMetric() {
        if (secondaryCMet != null) {
            return secondaryCMet;
        } else {
            return primaryCMet;
        }
    }

    /**
     * This method returns the current NeighborSetFinder.
     *
     * @return NeighborSet finder that is currently in use by the system.
     */
    private NeighborSetFinder getNSF() {
        if (nsfSecondary != null) {
            return nsfSecondary;
        } else {
            return nsfPrimary;
        }
    }

    /**
     * Creates new form HubExplorer
     */
    public ImageHubExplorer() {
        initComponents();
        additionalInit();
    }

    /**
     * Initialization.
     */
    private void additionalInit() {
        // Initialize kNN and reverse neighbor panels.
        nnPanel.setLayout(new FlowLayout());
        rnnPanel.setLayout(new FlowLayout());
        nnPanel.setComponentOrientation(
                ComponentOrientation.LEFT_TO_RIGHT);
        rnnPanel.setComponentOrientation(
                ComponentOrientation.LEFT_TO_RIGHT);
        rnnPanel.setMaximumSize(new Dimension(60000, 400));
        // Initialize the neighborhood size selection slider.
        kSelectionSlider.setMajorTickSpacing(5);
        kSelectionSlider.setMinorTickSpacing(1);
        kSelectionSlider.setPaintLabels(true);
        kSelectionSlider.setPaintTicks(true);
        kSelectionSlider.addChangeListener(new SliderChanger());
        // Initialize various chart panels.
        classColorAndNamesPanel.setLayout(new VerticalFlowLayout());
        occProfileChartHolder.setLayout(new FlowLayout());
        classDistributionHolder.setLayout(new FlowLayout());
        chartHoldingPanelOccDistribution.setLayout(new FlowLayout());
        chartHoldingPanelOccDistribution.setPreferredSize(
                new Dimension(497, 191));
        queryNNPanel.setLayout(new VerticalFlowLayout());
        classifierPredictionsPanel.setLayout(new VerticalFlowLayout());
        // Initialize the scroll panes.
        prClassScrollPane.setPreferredSize(new Dimension(237, 432));
        prClassScrollPane.setMinimumSize(new Dimension(237, 432));
        prClassScrollPane.setMaximumSize(new Dimension(237, 432));
        queryNNScrollPane.setPreferredSize(new Dimension(207, 455));
        queryNNScrollPane.setMinimumSize(new Dimension(207, 455));
        queryNNScrollPane.setMaximumSize(new Dimension(207, 455));
        classesScrollPanel.setLayout(new VerticalFlowLayout());
        classesScrollPanel.setPreferredSize(new Dimension(760, 1168));
        classesScrollPanel.setMaximumSize(new Dimension(760, 1168));
        classesScrollPanel.setMinimumSize(new Dimension(760, 1168));
        classesScrollPane.setPreferredSize(new Dimension(760, 308));
        classesScrollPane.setMaximumSize(new Dimension(760, 308));
        classesScrollPane.setMinimumSize(new Dimension(760, 308));
        // Initialize the MDS component.
        mdsCollectionPanel.setPreferredSize(new Dimension(1500, 1500));
        mdsCollectionPanel.setMaximumSize(new Dimension(1500, 1500));
        mdsCollectionPanel.setMinimumSize(new Dimension(1500, 1500));
        // Initialize the kNN graph visualization component.
        neighborGraphScrollPane.setPreferredSize(new Dimension(550, 550));
        neighborGraphScrollPane.setMinimumSize(new Dimension(550, 550));
        neighborGraphScrollPane.setMaximumSize(new Dimension(550, 550));
        neighborGraphScrollPane.setVisible(true);
        selectedImagePathLabelClassNeighborMain.setPreferredSize(
                new Dimension(30, 16));
        selectedImagePathLabelClassNeighborMain.setMaximumSize(
                new Dimension(30, 16));
        selectedImagePathLabelClassNeighborMain.setMinimumSize(
                new Dimension(30, 16));
        jScrollPane1.setPreferredSize(new Dimension(30, 16));
        jScrollPane1.setMinimumSize(new Dimension(30, 16));
        jScrollPane1.setMaximumSize(new Dimension(30, 16));
    }

    /**
     * This listener handles the changes in the current neighborhood size by
     * moving the k-slider.
     */
    class SliderChanger implements ChangeListener {

        @Override
        public void stateChanged(ChangeEvent e) {
            if (!neighborStatsCalculated) {
                // If the kNN stats haven't been calculated, there is no need to
                // do anything.
                return;
            }
            Object src = e.getSource();
            if (src instanceof JSlider) {
                // Get the selected neighborhood size.
                neighborhoodSize = Math.max(((JSlider) src).getValue(), 1);
                // Get the index of the currently selected image.
                int index = -1;
                if (selectedImageHistory != null
                        && selectedImageHistory.size() > 0) {
                    index = selectedImageHistory.get(
                            selectedImageIndexInHistory);
                }
                // Get the object holding the kNN sets.
                NeighborSetFinder nsf = getNSF();
                // Reinitialize the panels.
                nnPanel.removeAll();
                rnnPanel.removeAll();
                nnPanel.revalidate();
                nnPanel.repaint();
                rnnPanel.revalidate();
                rnnPanel.repaint();
                if (index != -1) {
                    // Get the kNN sets.
                    int[][] kneighbors = nsf.getKNeighbors();
                    // Refresh the neighbor list.
                    for (int i = 0; i < neighborhoodSize; i++) {
                        ImagePanelWithClass neighbor =
                                new ImagePanelWithClass(classColors);
                        neighbor.addMouseListener(
                                new NeighborSelectionListener());
                        neighbor.setImage(
                                thumbnails.get(kneighbors[index][i]),
                                quantizedRepresentation.getLabelOf(
                                kneighbors[index][i]), kneighbors[index][i]);
                        nnPanel.add(neighbor);
                    }
                    // Refresh the reverse nearest neighbor list.
                    ArrayList<Integer>[] rrns =
                            rnnSetsAllK[neighborhoodSize - 1];
                    if (rrns[index] != null && rrns[index].size() > 0) {
                        for (int i = 0; i < rrns[index].size(); i++) {
                            ImagePanelWithClass rrneighbor =
                                    new ImagePanelWithClass(classColors);
                            rrneighbor.setImage(
                                    thumbnails.get(rrns[index].get(i)),
                                    quantizedRepresentation.getLabelOf(
                                    rrns[index].get(i)), rrns[index].get(i));
                            rrneighbor.addMouseListener(
                                    new NeighborSelectionListener());
                            rnnPanel.add(rrneighbor);
                        }
                    }
                    // Refresh the occurrence profile of the current image.
                    DefaultPieDataset pieData = new DefaultPieDataset();
                    for (int c = 0; c < numClasses; c++) {
                        pieData.setValue(classNames[c],
                                occurrenceProfilesAllK[neighborhoodSize - 1][
                                index][c]);
                    }
                    JFreeChart chart = ChartFactory.createPieChart3D(
                            "occurrence profile", pieData, true, true, false);
                    PiePlot3D plot = (PiePlot3D) chart.getPlot();
                    plot.setStartAngle(290);
                    plot.setDirection(Rotation.CLOCKWISE);
                    plot.setForegroundAlpha(0.5f);
                    PieRenderer prend = new PieRenderer(classColors);
                    prend.setColor(plot, pieData);
                    ChartPanel chartPanel = new ChartPanel(chart);
                    chartPanel.setPreferredSize(new Dimension(240, 200));
                    // Refresh the display.
                    occProfileChartHolder.removeAll();
                    occProfileChartHolder.add(chartPanel);
                    occProfileChartHolder.revalidate();
                    occProfileChartHolder.repaint();
                    nnPanel.revalidate();
                    nnPanel.repaint();
                    rnnPanel.revalidate();
                    rnnPanel.repaint();
                    // Refresh the kNN graph visualizations, as new edges might
                    // need to be inserted.
                    graphVServers[neighborhoodSize - 1].setPreferredSize(
                            new Dimension(500, 500));
                    graphVServers[neighborhoodSize - 1].setMinimumSize(
                            new Dimension(500, 500));
                    Layout<ImageNode, NeighborLink> layout =
                            new CircleLayout(
                            neighborGraphs[neighborhoodSize - 1]);
                    layout.setSize(new Dimension(500, 500));
                    layout.initialize();
                    VisualizationViewer<ImageNode, NeighborLink> vv =
                            new VisualizationViewer<>(layout);
                    vv.setPreferredSize(new Dimension(550, 550));
                    vv.setMinimumSize(new Dimension(550, 550));
                    vv.setDoubleBuffered(true);
                    vv.setEnabled(true);
                    graphVServers[neighborhoodSize - 1] = vv;
                    vv.getRenderContext().setVertexIconTransformer(
                            new IconTransformer<ImageNode, Icon>());
                    vv.getRenderContext().setVertexShapeTransformer(
                            new ShapeTransformer<ImageNode, Shape>());
                    vv.getRenderContext().setEdgeArrowPredicate(
                            new DirectionDisplayPredicate());
                    vv.getRenderContext().setEdgeLabelTransformer(
                            new Transformer() {
                        @Override
                        public String transform(Object e) {
                            return (e.toString());
                        }
                    });
                    PluggableGraphMouse gm = new PluggableGraphMouse();
                    gm.add(new PickingGraphMousePlugin());
                    vv.setGraphMouse(gm);
                    vv.setBackground(Color.WHITE);
                    vv.setVisible(true);
                    final PickedState<ImageNode> pickedState =
                            vv.getPickedVertexState();
                    pickedState.addItemListener(new ItemListener() {
                        @Override
                        public void itemStateChanged(ItemEvent e) {
                            Object subject = e.getItem();
                            if (subject instanceof ImageNode) {
                                ImageNode vertex = (ImageNode) subject;
                                if (pickedState.isPicked(vertex)) {
                                    setSelectedImageForIndex(vertex.id);
                                } else {
                                }
                            }
                        }
                    });
                    // Refresh the graph displays.
                    vv.validate();
                    neighborGraphScrollPane.setViewportView(
                            graphVServers[neighborhoodSize - 1]);
                    neighborGraphScrollPane.revalidate();
                    neighborGraphScrollPane.repaint();
                }
                // Refresh the kNN stats in the main screen.
                setStatFieldsForK(neighborhoodSize);
                classesScrollPanel.removeAll();
                // Refresh the class summary panels
                for (int c = 0; c < numClasses; c++) {
                    ClassHubsPanel chp =
                            new ClassHubsPanel(classColors[c], classNames[c]);
                    classStatsOverviews[c] = chp;
                    chp.setPointTypeDistribution(classPTypes[c]);
                    chp.revalidate();
                    chp.repaint();
                    // Lists of hubs, good hubs and bad hubs for the class.
                    JPanel hubsPanel = chp.getHubsPanel();
                    JPanel hubsPanelGood = chp.getGoodHubsPanel();
                    JPanel hubsPanelBad = chp.getBadHubsPanel();
                    hubsPanel.removeAll();
                    for (int i = 0; i < Math.min(
                            50, classImageIndexes[c].size()); i++) {
                        BufferedImage thumb = thumbnails.get(
                                classTopHubLists[
                                neighborhoodSize - 1][c].get(i));
                        ImagePanelWithClass imgPan =
                                new ImagePanelWithClass(classColors);
                        imgPan.addMouseListener(
                                new NeighborSelectionListener());
                        imgPan.setImage(thumb,
                                quantizedRepresentation.getLabelOf(
                                classTopHubLists[neighborhoodSize - 1][c].
                                get(i)), classTopHubLists[neighborhoodSize - 1][
                                c].get(i));
                        hubsPanel.add(imgPan);
                    }
                    hubsPanel.revalidate();
                    hubsPanel.repaint();
                    hubsPanelGood.removeAll();
                    for (int i = 0; i < Math.min(50,
                            classImageIndexes[c].size()); i++) {
                        BufferedImage thumb = thumbnails.get(
                                classTopGoodHubsList[neighborhoodSize - 1][c].
                                get(i));
                        ImagePanelWithClass imgPan =
                                new ImagePanelWithClass(classColors);
                        imgPan.addMouseListener(
                                new NeighborSelectionListener());
                        imgPan.setImage(thumb,
                                quantizedRepresentation.getLabelOf(
                                classTopGoodHubsList[neighborhoodSize - 1][c].
                                get(i)), classTopGoodHubsList[
                                neighborhoodSize - 1][c].get(i));
                        hubsPanelGood.add(imgPan);
                    }
                    hubsPanelGood.revalidate();
                    hubsPanelGood.repaint();
                    hubsPanelBad.removeAll();
                    for (int i = 0; i < Math.min(50,
                            classImageIndexes[c].size()); i++) {
                        BufferedImage thumb = thumbnails.get(
                                classTopBadHubsList[neighborhoodSize - 1][c].
                                get(i));
                        ImagePanelWithClass imgPan =
                                new ImagePanelWithClass(classColors);
                        imgPan.addMouseListener(
                                new NeighborSelectionListener());
                        imgPan.setImage(thumb,
                                quantizedRepresentation.getLabelOf(
                                classTopBadHubsList[neighborhoodSize - 1][c].
                                get(i)), classTopBadHubsList[
                                neighborhoodSize - 1][c].get(i));
                        hubsPanelBad.add(imgPan);
                    }
                    hubsPanelBad.revalidate();
                    hubsPanelBad.repaint();
                    chp.revalidate();
                    chp.repaint();
                    classesScrollPanel.add(chp);
                }
                classesScrollPanel.revalidate();
                classesScrollPanel.repaint();
                classesScrollPane.revalidate();
                classesScrollPane.repaint();
                // Handle the data visualization in the MDS screen.
                if (imageCoordinatesXY != null) {
                    // In case some of the thumbnails crosses the bounding box
                    // of the MDS panel, offsets are set to compensate and to
                    // ensure all images are visible in their entirety.
                    float offX, offY;
                    if (highestHubnesses != null) {
                        float maxOccurrenceFrequency =
                                ArrayUtil.max(highestHubnesses[
                                neighborhoodSize - 1]);
                        float[] thumbSizes = new float[highestHubnesses[
                                neighborhoodSize - 1].length];
                        ArrayList<Rectangle2D> bounds =
                                new ArrayList<>(thumbSizes.length);
                        ArrayList<ImagePanelWithClass> imgsMDS =
                                new ArrayList<>(thumbSizes.length);
                        for (int i = 0; i < thumbSizes.length; i++) {
                            // Calculate the thumbnail size.
                            try {
                                thumbSizes[i] = pointScale(
                                        highestHubnesses[
                                        neighborhoodSize - 1][i],
                                        maxOccurrenceFrequency,
                                        minImageScale,
                                        maxImageScale);
                            } catch (Exception eSecond) {
                                System.err.println(eSecond.getMessage());
                            }
                            if (imageCoordinatesXY[
                                    highestHubIndexes[
                                    neighborhoodSize - 1][i]][0]
                                    + thumbSizes[i] / 2
                                    > mdsCollectionPanel.getWidth()) {
                                offX = (thumbSizes[i] / 2
                                        - (mdsCollectionPanel.getWidth()
                                        - imageCoordinatesXY[highestHubIndexes[
                                        neighborhoodSize - 1][i]][0]));
                            } else if (imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][0]
                                    - thumbSizes[i] / 2 < 0) {
                                offX = imageCoordinatesXY[highestHubIndexes[
                                        neighborhoodSize - 1][i]][0];
                            } else {
                                offX = thumbSizes[i] / 2;
                            }
                            if (imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][1]
                                    + thumbSizes[i] / 2
                                    > mdsCollectionPanel.getHeight()) {
                                offY = (thumbSizes[i] / 2
                                        - (mdsCollectionPanel.getHeight()
                                        - imageCoordinatesXY[highestHubIndexes[
                                        neighborhoodSize - 1][i]][1]));
                            } else if (imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][1]
                                    - thumbSizes[i] / 2 < 0) {
                                offY = imageCoordinatesXY[highestHubIndexes[
                                        neighborhoodSize - 1][i]][1];
                            } else {
                                offY = thumbSizes[i] / 2;
                            }
                            // Get the image thumbnail to show.
                            BufferedImage thumb = thumbnails.get(
                                    highestHubIndexes[neighborhoodSize - 1][i]);
                            ImagePanelWithClass imgPan =
                                    new ImagePanelWithClass(classColors);
                            imgPan.addMouseListener(
                                    new NeighborSelectionListener());
                            imgPan.setImage(thumb,
                                    quantizedRepresentation.getLabelOf(
                                    highestHubIndexes[neighborhoodSize - 1][i]),
                                    highestHubIndexes[neighborhoodSize - 1][i]);
                            imgsMDS.add(imgPan);
                            // Set the bounding rectangle.
                            bounds.add(new Rectangle2D.Float(
                                    imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][0] - offX,
                                    imageCoordinatesXY[highestHubIndexes[
                                    neighborhoodSize - 1][i]][1] - offY,
                                    thumbSizes[i], thumbSizes[i]));
                        }
                        // Set the images for display in the MDS overview panel.
                        mdsCollectionPanel.setImageSet(imgsMDS, bounds);
                        setMDSBackground();
                        // Refresh the display.
                        mdsCollectionPanel.revalidate();
                        mdsCollectionPanel.repaint();
                    }
                }
            }
        }
    }

    /**
     * This method loads the distance matrix from a file.
     *
     * @param dMatFile File that holds the distance matrix.
     * @return float[][] that is the distance matrix.
     * @throws Exception
     */
    public float[][] loadDMatFromFile(File dMatFile) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(dMatFile)));
        float[][] dMatLoaded = null;
        String line;
        String[] lineItems;
        try {
            int size = Integer.parseInt(br.readLine());
            dMatLoaded = new float[size][];
            for (int i = 0; i < size - 1; i++) {
                dMatLoaded[i] = new float[size - i - 1];
                line = br.readLine();
                lineItems = line.split(",");
                for (int j = 0; j < lineItems.length; j++) {
                    dMatLoaded[i][j] = Float.parseFloat(lineItems[j]);
                }
            }
            dMatLoaded[size - 1] = new float[0];
        } catch (IOException | NumberFormatException e) {
            throw e;
        } finally {
            br.close();
        }
        return dMatLoaded;
    }

    /**
     * This method prints a distance matrix to a file.
     *
     * @param distMat float[][] that is the distance matrix.
     * @param dMatFile File to write the matrix to.
     * @throws Exception
     */
    public void printDMatToFile(float[][] distMat, File dMatFile)
            throws Exception {
        FileUtil.createFile(dMatFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(dMatFile));) {
            pw.println(distMat.length);
            for (int i = 0; i < distMat.length - 1; i++) {
                pw.print(distMat[i][0]);
                for (int j = 1; j < distMat[i].length; j++) {
                    pw.print("," + distMat[i][j]);
                }
                pw.println();
            }
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        hubTab = new javax.swing.JTabbedPane();
        dataMainPanel = new javax.swing.JPanel();
        selectedImagePanelClassNeighborMain = new gui.images.ImagePanel();
        selectedImageLabelClassNeighborMain = new javax.swing.JLabel();
        mdsScrollPane = new javax.swing.JScrollPane();
        mdsCollectionPanel = new gui.images.ImagesDisplayPanel();
        workspaceLabelTxt = new javax.swing.JLabel();
        collectionSizeLabelTxt = new javax.swing.JLabel();
        workspaceLabelValue = new javax.swing.JLabel();
        collectionSizeLabelValue = new javax.swing.JLabel();
        kSelectionSlider = new javax.swing.JSlider();
        nhSizeLabelTxt = new javax.swing.JLabel();
        numClassesLabelTxt = new javax.swing.JLabel();
        numClassesLabelValue = new javax.swing.JLabel();
        hRelatedPropTxt = new javax.swing.JLabel();
        skewnwessLabelTxt = new javax.swing.JLabel();
        skewnessLabelValue = new javax.swing.JLabel();
        kurtosisLabelTxt = new javax.swing.JLabel();
        kurtosisLabelValue = new javax.swing.JLabel();
        nkEntropyLabelTxt = new javax.swing.JLabel();
        nkEntropyLabelValue = new javax.swing.JLabel();
        jLabel1 = new javax.swing.JLabel();
        rnkEntropyValue = new javax.swing.JLabel();
        nkEntropySkewnessTxt = new javax.swing.JLabel();
        rnkEntropySkewnessTxt = new javax.swing.JLabel();
        nkEntropySkewnessValues = new javax.swing.JLabel();
        rnkEntropySkewnessValue = new javax.swing.JLabel();
        percAboveLabelTxt = new javax.swing.JLabel();
        percAboveLabelValue = new javax.swing.JLabel();
        hubsLabelTxt = new javax.swing.JLabel();
        orphansLabelTxt = new javax.swing.JLabel();
        regularLabelTxt = new javax.swing.JLabel();
        majorDegLabelTxt = new javax.swing.JLabel();
        hubsLabelValue = new javax.swing.JLabel();
        orphansLabelValue = new javax.swing.JLabel();
        regularLabelValue = new javax.swing.JLabel();
        majorDegLabelValue = new javax.swing.JLabel();
        badHubnessLabelTxt = new javax.swing.JLabel();
        badHubnessLabelValue = new javax.swing.JLabel();
        chartHoldingPanelOccDistribution = new javax.swing.JPanel();
        jScrollPane1 = new javax.swing.JScrollPane();
        selectedImagePathLabelClassNeighborMain = new javax.swing.JLabel();
        neighborPanel = new javax.swing.JPanel();
        selectedImagePanelClassNeighbor = new gui.images.ImagePanel();
        selectedImageLabelClassNeighbor = new javax.swing.JLabel();
        nnScrollPane = new javax.swing.JScrollPane();
        nnPanel = new javax.swing.JPanel();
        rnnScrollPane = new javax.swing.JScrollPane();
        rnnPanel = new javax.swing.JPanel();
        nnScrollLabelTxt = new javax.swing.JLabel();
        rnnScrollLabelTxt = new javax.swing.JLabel();
        occProfileChartHolder = new javax.swing.JPanel();
        noccProfLabelTxt = new javax.swing.JLabel();
        neighborGraphScrollPane = new javax.swing.JScrollPane();
        addSelectedButton = new javax.swing.JButton();
        addNNsButton = new javax.swing.JButton();
        addRNNsButton = new javax.swing.JButton();
        jScrollPane2 = new javax.swing.JScrollPane();
        selectedImagePathLabelClassNeighbor = new javax.swing.JLabel();
        removeVertexButton = new javax.swing.JButton();
        removeAllButton = new javax.swing.JButton();
        classPanel = new javax.swing.JPanel();
        selectedImagePanelClass = new gui.images.ImagePanel();
        selectedImageLabelClass = new javax.swing.JLabel();
        confusionMatScrollPane = new javax.swing.JScrollPane();
        classHubnessTable = new javax.swing.JTable();
        classesScrollPane = new javax.swing.JScrollPane();
        classesScrollPanel = new javax.swing.JPanel();
        classDistributionHolder = new javax.swing.JPanel();
        jScrollPane3 = new javax.swing.JScrollPane();
        selectedImagePathLabelClass = new javax.swing.JLabel();
        cNamesScrollPane = new javax.swing.JScrollPane();
        jScrollPane5 = new javax.swing.JScrollPane();
        classColorAndNamesPanel = new javax.swing.JPanel();
        searchPanel = new javax.swing.JPanel();
        selectedImagePanelSearch = new gui.images.ImagePanel();
        selectedImageLabelSearch = new javax.swing.JLabel();
        searchQLabelTxt = new javax.swing.JLabel();
        queryImagePanel = new gui.images.ImagePanel();
        imageBrowseButton = new javax.swing.JButton();
        jTextField1 = new javax.swing.JTextField();
        queryQTextLabelTxt = new javax.swing.JLabel();
        queryNNScrollPane = new javax.swing.JScrollPane();
        queryNNPanel = new javax.swing.JPanel();
        simResLabelTxt = new javax.swing.JLabel();
        searchButton = new javax.swing.JButton();
        prClassLabelTxt = new javax.swing.JLabel();
        prClassScrollPane = new javax.swing.JScrollPane();
        classifierPredictionsPanel = new javax.swing.JPanel();
        collectionSearchButton = new javax.swing.JButton();
        jScrollPane4 = new javax.swing.JScrollPane();
        selectedImagePathLabelSearch = new javax.swing.JLabel();
        reRankingButton = new javax.swing.JButton();
        menuBar = new javax.swing.JMenuBar();
        collectionMenu = new javax.swing.JMenu();
        workspaceMenuItem = new javax.swing.JMenuItem();
        importItem = new javax.swing.JMenuItem();
        dMatrixMenu = new javax.swing.JMenu();
        distImportItem = new javax.swing.JMenuItem();
        distCalculateMenu = new javax.swing.JMenu();
        manhattanDistItem = new javax.swing.JMenuItem();
        distCalcEuclideanItem = new javax.swing.JMenuItem();
        distCalcCosineItem = new javax.swing.JMenuItem();
        tanimotoMenuItem = new javax.swing.JMenuItem();
        klMenuItem = new javax.swing.JMenuItem();
        bcMenuItem = new javax.swing.JMenuItem();
        canMenuItem = new javax.swing.JMenuItem();
        neighborStatsItem = new javax.swing.JMenuItem();
        mdsVisualizeItem = new javax.swing.JMenuItem();
        selImgPathMenuItem = new javax.swing.JMenuItem();
        majorHubSelectionItem = new javax.swing.JMenuItem();
        metricLearningMenu = new javax.swing.JMenu();
        secondaryMetricMenu = new javax.swing.JMenu();
        simcosMenuItem = new javax.swing.JMenuItem();
        simhubMenuItem = new javax.swing.JMenuItem();
        mpMenuItem = new javax.swing.JMenuItem();
        localScalingItem = new javax.swing.JMenuItem();
        nicdmItem = new javax.swing.JMenuItem();
        loadSecondaryDistancesItem = new javax.swing.JMenuItem();
        editMenu = new javax.swing.JMenu();
        previousMenuItem = new javax.swing.JMenuItem();
        nextMenuItem = new javax.swing.JMenuItem();
        screenCaptureMenu = new javax.swing.JMenu();
        mdsScreenCaptureItem = new javax.swing.JMenuItem();
        graphScreenCaptureItem = new javax.swing.JMenuItem();
        codebookMenu = new javax.swing.JMenu();
        loadCodebookItem = new javax.swing.JMenuItem();
        loadCodebookProfileMenuItem = new javax.swing.JMenuItem();
        classificationMenu = new javax.swing.JMenu();
        trainModelsItem = new javax.swing.JMenuItem();
        selImageMenu = new javax.swing.JMenu();
        selSIFTmenuItem = new javax.swing.JMenuItem();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Image Hub Explorer");

        javax.swing.GroupLayout selectedImagePanelClassNeighborMainLayout = new javax.swing.GroupLayout(selectedImagePanelClassNeighborMain);
        selectedImagePanelClassNeighborMain.setLayout(selectedImagePanelClassNeighborMainLayout);
        selectedImagePanelClassNeighborMainLayout.setHorizontalGroup(
            selectedImagePanelClassNeighborMainLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 241, Short.MAX_VALUE)
        );
        selectedImagePanelClassNeighborMainLayout.setVerticalGroup(
            selectedImagePanelClassNeighborMainLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 257, Short.MAX_VALUE)
        );

        selectedImageLabelClassNeighborMain.setText("Current Image");

        mdsScrollPane.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        mdsScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        mdsCollectionPanel.setName(""); // NOI18N
        mdsCollectionPanel.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                mdsCollectionPanelMouseClicked(evt);
            }
        });

        javax.swing.GroupLayout mdsCollectionPanelLayout = new javax.swing.GroupLayout(mdsCollectionPanel);
        mdsCollectionPanel.setLayout(mdsCollectionPanelLayout);
        mdsCollectionPanelLayout.setHorizontalGroup(
            mdsCollectionPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 2000, Short.MAX_VALUE)
        );
        mdsCollectionPanelLayout.setVerticalGroup(
            mdsCollectionPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 2000, Short.MAX_VALUE)
        );

        mdsScrollPane.setViewportView(mdsCollectionPanel);

        workspaceLabelTxt.setText("Workspace:");

        collectionSizeLabelTxt.setText("Collection size:");

        workspaceLabelValue.setText("...");

        collectionSizeLabelValue.setText("...");

        kSelectionSlider.setMaximum(50);
        kSelectionSlider.setPaintLabels(true);
        kSelectionSlider.setPaintTicks(true);
        kSelectionSlider.setToolTipText("Select the neighborhood size, k");
        kSelectionSlider.setValue(1);

        nhSizeLabelTxt.setText("Neighborhood size (k):");

        numClassesLabelTxt.setText("Num. Classes:");

        numClassesLabelValue.setText("...");

        hRelatedPropTxt.setBackground(new java.awt.Color(102, 153, 255));
        hRelatedPropTxt.setText("Hubness-related properties:");

        skewnwessLabelTxt.setText("Occ. Skewness: ");

        skewnessLabelValue.setText("...");

        kurtosisLabelTxt.setText("Occ. Kurtosis:");

        kurtosisLabelValue.setText("...");

        nkEntropyLabelTxt.setText("Nk Entropy:");

        nkEntropyLabelValue.setText("...");

        jLabel1.setText("RNk Entropy:");

        rnkEntropyValue.setText("...");

        nkEntropySkewnessTxt.setText("Nk Ent. Skew:");

        rnkEntropySkewnessTxt.setText("RNk Ent Skew:");

        nkEntropySkewnessValues.setText("...");

        rnkEntropySkewnessValue.setText("...");

        percAboveLabelTxt.setText("Perc Nk(x) > 0");

        percAboveLabelValue.setText("...");

        hubsLabelTxt.setText("Hubs:");

        orphansLabelTxt.setText("Orphans:");

        regularLabelTxt.setText("Regular:");

        majorDegLabelTxt.setText("Major Deg:");

        hubsLabelValue.setText("...");

        orphansLabelValue.setText("...");

        regularLabelValue.setText("...");

        majorDegLabelValue.setText("...");

        badHubnessLabelTxt.setText("Mislabel perc:");

        badHubnessLabelValue.setText("...");

        chartHoldingPanelOccDistribution.setMaximumSize(new java.awt.Dimension(497, 191));
        chartHoldingPanelOccDistribution.setMinimumSize(new java.awt.Dimension(497, 191));

        javax.swing.GroupLayout chartHoldingPanelOccDistributionLayout = new javax.swing.GroupLayout(chartHoldingPanelOccDistribution);
        chartHoldingPanelOccDistribution.setLayout(chartHoldingPanelOccDistributionLayout);
        chartHoldingPanelOccDistributionLayout.setHorizontalGroup(
            chartHoldingPanelOccDistributionLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 497, Short.MAX_VALUE)
        );
        chartHoldingPanelOccDistributionLayout.setVerticalGroup(
            chartHoldingPanelOccDistributionLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 221, Short.MAX_VALUE)
        );

        jScrollPane1.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        jScrollPane1.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        selectedImagePathLabelClassNeighborMain.setText("Path:");
        jScrollPane1.setViewportView(selectedImagePathLabelClassNeighborMain);

        javax.swing.GroupLayout dataMainPanelLayout = new javax.swing.GroupLayout(dataMainPanel);
        dataMainPanel.setLayout(dataMainPanelLayout);
        dataMainPanelLayout.setHorizontalGroup(
            dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, dataMainPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(mdsScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 536, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(dataMainPanelLayout.createSequentialGroup()
                        .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(rnkEntropySkewnessTxt)
                            .addComponent(nkEntropySkewnessTxt)
                            .addComponent(jLabel1)
                            .addComponent(nkEntropyLabelTxt)
                            .addComponent(hRelatedPropTxt, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addGroup(dataMainPanelLayout.createSequentialGroup()
                                .addComponent(workspaceLabelTxt)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(workspaceLabelValue, javax.swing.GroupLayout.PREFERRED_SIZE, 169, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(dataMainPanelLayout.createSequentialGroup()
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                    .addComponent(numClassesLabelTxt, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                    .addComponent(collectionSizeLabelTxt, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addComponent(numClassesLabelValue, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                    .addComponent(collectionSizeLabelValue, javax.swing.GroupLayout.DEFAULT_SIZE, 68, Short.MAX_VALUE)))
                            .addComponent(kSelectionSlider, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, 249, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(dataMainPanelLayout.createSequentialGroup()
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                    .addComponent(kurtosisLabelTxt, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                    .addComponent(skewnwessLabelTxt, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(nkEntropySkewnessValues)
                                    .addGroup(dataMainPanelLayout.createSequentialGroup()
                                        .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addComponent(skewnessLabelValue)
                                            .addComponent(kurtosisLabelValue)
                                            .addComponent(nkEntropyLabelValue)
                                            .addComponent(rnkEntropyValue)
                                            .addComponent(rnkEntropySkewnessValue))
                                        .addGap(28, 28, 28)
                                        .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addGroup(dataMainPanelLayout.createSequentialGroup()
                                                .addComponent(badHubnessLabelTxt)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                                .addComponent(badHubnessLabelValue))
                                            .addGroup(dataMainPanelLayout.createSequentialGroup()
                                                .addComponent(majorDegLabelTxt)
                                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                                .addComponent(majorDegLabelValue))
                                            .addGroup(dataMainPanelLayout.createSequentialGroup()
                                                .addComponent(regularLabelTxt)
                                                .addGap(18, 18, 18)
                                                .addComponent(regularLabelValue))
                                            .addGroup(dataMainPanelLayout.createSequentialGroup()
                                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                                    .addComponent(hubsLabelTxt)
                                                    .addComponent(orphansLabelTxt))
                                                .addGap(18, 18, 18)
                                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                                    .addComponent(orphansLabelValue)
                                                    .addComponent(hubsLabelValue)))))))
                            .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                .addGroup(dataMainPanelLayout.createSequentialGroup()
                                    .addComponent(percAboveLabelTxt)
                                    .addGap(18, 18, 18)
                                    .addComponent(percAboveLabelValue, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                .addComponent(nhSizeLabelTxt, javax.swing.GroupLayout.Alignment.LEADING)))
                        .addGap(7, 7, 7)
                        .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(selectedImageLabelClassNeighborMain, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(selectedImagePanelClassNeighborMain, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 241, javax.swing.GroupLayout.PREFERRED_SIZE)))
                    .addComponent(chartHoldingPanelOccDistribution, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap())
        );
        dataMainPanelLayout.setVerticalGroup(
            dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(dataMainPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(mdsScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 593, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(dataMainPanelLayout.createSequentialGroup()
                        .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(dataMainPanelLayout.createSequentialGroup()
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(workspaceLabelTxt)
                                    .addComponent(workspaceLabelValue))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addComponent(collectionSizeLabelTxt, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                    .addComponent(collectionSizeLabelValue, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                .addGap(19, 19, 19)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                    .addComponent(numClassesLabelTxt, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                    .addComponent(numClassesLabelValue, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                .addGap(17, 17, 17)
                                .addComponent(hRelatedPropTxt)
                                .addGap(15, 15, 15)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(skewnwessLabelTxt)
                                    .addComponent(skewnessLabelValue)
                                    .addComponent(hubsLabelTxt)
                                    .addComponent(hubsLabelValue))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(kurtosisLabelTxt)
                                    .addComponent(kurtosisLabelValue)
                                    .addComponent(orphansLabelTxt)
                                    .addComponent(orphansLabelValue))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(nkEntropyLabelTxt)
                                    .addComponent(nkEntropyLabelValue)
                                    .addComponent(regularLabelTxt)
                                    .addComponent(regularLabelValue))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(jLabel1)
                                    .addComponent(rnkEntropyValue)
                                    .addComponent(majorDegLabelTxt)
                                    .addComponent(majorDegLabelValue))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(nkEntropySkewnessTxt)
                                    .addComponent(nkEntropySkewnessValues))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(rnkEntropySkewnessTxt)
                                    .addComponent(rnkEntropySkewnessValue)
                                    .addComponent(badHubnessLabelTxt)
                                    .addComponent(badHubnessLabelValue))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(percAboveLabelTxt)
                                    .addComponent(percAboveLabelValue)))
                            .addComponent(selectedImagePanelClassNeighborMain, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(selectedImageLabelClassNeighborMain, javax.swing.GroupLayout.DEFAULT_SIZE, 53, Short.MAX_VALUE)
                            .addComponent(nhSizeLabelTxt, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(dataMainPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(kSelectionSlider, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 53, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(chartHoldingPanelOccDistribution, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addContainerGap())
        );

        hubTab.addTab("Data Overview", dataMainPanel);

        javax.swing.GroupLayout selectedImagePanelClassNeighborLayout = new javax.swing.GroupLayout(selectedImagePanelClassNeighbor);
        selectedImagePanelClassNeighbor.setLayout(selectedImagePanelClassNeighborLayout);
        selectedImagePanelClassNeighborLayout.setHorizontalGroup(
            selectedImagePanelClassNeighborLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 222, Short.MAX_VALUE)
        );
        selectedImagePanelClassNeighborLayout.setVerticalGroup(
            selectedImagePanelClassNeighborLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 224, Short.MAX_VALUE)
        );

        selectedImageLabelClassNeighbor.setText("Current Image");

        nnScrollPane.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        nnScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        javax.swing.GroupLayout nnPanelLayout = new javax.swing.GroupLayout(nnPanel);
        nnPanel.setLayout(nnPanelLayout);
        nnPanelLayout.setHorizontalGroup(
            nnPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 353, Short.MAX_VALUE)
        );
        nnPanelLayout.setVerticalGroup(
            nnPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 100, Short.MAX_VALUE)
        );

        nnScrollPane.setViewportView(nnPanel);

        rnnScrollPane.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        rnnScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        javax.swing.GroupLayout rnnPanelLayout = new javax.swing.GroupLayout(rnnPanel);
        rnnPanel.setLayout(rnnPanelLayout);
        rnnPanelLayout.setHorizontalGroup(
            rnnPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 373, Short.MAX_VALUE)
        );
        rnnPanelLayout.setVerticalGroup(
            rnnPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 100, Short.MAX_VALUE)
        );

        rnnScrollPane.setViewportView(rnnPanel);

        nnScrollLabelTxt.setFont(new java.awt.Font("Tahoma", 0, 18)); // NOI18N
        nnScrollLabelTxt.setText("NNs:");

        rnnScrollLabelTxt.setFont(new java.awt.Font("Tahoma", 0, 18)); // NOI18N
        rnnScrollLabelTxt.setText("RNNs:");

        javax.swing.GroupLayout occProfileChartHolderLayout = new javax.swing.GroupLayout(occProfileChartHolder);
        occProfileChartHolder.setLayout(occProfileChartHolderLayout);
        occProfileChartHolderLayout.setHorizontalGroup(
            occProfileChartHolderLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 236, Short.MAX_VALUE)
        );
        occProfileChartHolderLayout.setVerticalGroup(
            occProfileChartHolderLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 185, Short.MAX_VALUE)
        );

        noccProfLabelTxt.setText("Selected image neighbor occurrence profile");

        neighborGraphScrollPane.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        neighborGraphScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        addSelectedButton.setText("Add selected");
        addSelectedButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                addSelectedButtonActionPerformed(evt);
            }
        });

        addNNsButton.setText("Add NNs");
        addNNsButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                addNNsButtonActionPerformed(evt);
            }
        });

        addRNNsButton.setText("Add RNNs");
        addRNNsButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                addRNNsButtonActionPerformed(evt);
            }
        });

        selectedImagePathLabelClassNeighbor.setText("Path:");
        jScrollPane2.setViewportView(selectedImagePathLabelClassNeighbor);

        removeVertexButton.setText("Remove Sel.");
        removeVertexButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                removeVertexButtonActionPerformed(evt);
            }
        });

        removeAllButton.setText("Remove All");
        removeAllButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                removeAllButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout neighborPanelLayout = new javax.swing.GroupLayout(neighborPanel);
        neighborPanel.setLayout(neighborPanelLayout);
        neighborPanelLayout.setHorizontalGroup(
            neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(neighborPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(neighborGraphScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 529, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(neighborPanelLayout.createSequentialGroup()
                        .addGap(31, 31, 31)
                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(nnScrollLabelTxt, javax.swing.GroupLayout.PREFERRED_SIZE, 46, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(rnnScrollLabelTxt, javax.swing.GroupLayout.PREFERRED_SIZE, 68, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(44, 44, 44)
                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(nnScrollPane, 0, 0, Short.MAX_VALUE)
                            .addComponent(rnnScrollPane, javax.swing.GroupLayout.DEFAULT_SIZE, 361, Short.MAX_VALUE)))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, neighborPanelLayout.createSequentialGroup()
                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addGroup(javax.swing.GroupLayout.Alignment.LEADING, neighborPanelLayout.createSequentialGroup()
                                .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                    .addComponent(occProfileChartHolder, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(noccProfLabelTxt, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.PREFERRED_SIZE, 276, javax.swing.GroupLayout.PREFERRED_SIZE))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                            .addGroup(neighborPanelLayout.createSequentialGroup()
                                .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                    .addGroup(neighborPanelLayout.createSequentialGroup()
                                        .addComponent(addRNNsButton)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(addNNsButton))
                                    .addComponent(removeAllButton))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                    .addComponent(removeVertexButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                    .addComponent(addSelectedButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                                .addGap(18, 18, 18)))
                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(selectedImageLabelClassNeighbor, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(selectedImagePanelClassNeighbor, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jScrollPane2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
                .addContainerGap())
        );
        neighborPanelLayout.setVerticalGroup(
            neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(neighborPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(neighborGraphScrollPane, javax.swing.GroupLayout.DEFAULT_SIZE, 593, Short.MAX_VALUE)
                    .addGroup(neighborPanelLayout.createSequentialGroup()
                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(neighborPanelLayout.createSequentialGroup()
                                .addComponent(occProfileChartHolder, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(noccProfLabelTxt, javax.swing.GroupLayout.DEFAULT_SIZE, 36, Short.MAX_VALUE))
                            .addComponent(selectedImagePanelClassNeighbor, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addGroup(neighborPanelLayout.createSequentialGroup()
                                .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                    .addGroup(javax.swing.GroupLayout.Alignment.LEADING, neighborPanelLayout.createSequentialGroup()
                                        .addComponent(selectedImageLabelClassNeighbor)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 68, Short.MAX_VALUE))
                                    .addGroup(neighborPanelLayout.createSequentialGroup()
                                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                            .addComponent(removeVertexButton)
                                            .addComponent(removeAllButton))
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                            .addComponent(addSelectedButton, javax.swing.GroupLayout.DEFAULT_SIZE, 47, Short.MAX_VALUE)
                                            .addComponent(addNNsButton, javax.swing.GroupLayout.DEFAULT_SIZE, 47, Short.MAX_VALUE))))
                                .addGap(16, 16, 16))
                            .addGroup(neighborPanelLayout.createSequentialGroup()
                                .addComponent(addRNNsButton, javax.swing.GroupLayout.PREFERRED_SIZE, 45, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addGap(18, 18, 18)))
                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(nnScrollLabelTxt, javax.swing.GroupLayout.DEFAULT_SIZE, 125, Short.MAX_VALUE)
                            .addComponent(nnScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 115, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addGroup(neighborPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(rnnScrollLabelTxt, javax.swing.GroupLayout.DEFAULT_SIZE, 122, Short.MAX_VALUE)
                            .addComponent(rnnScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 111, javax.swing.GroupLayout.PREFERRED_SIZE))))
                .addContainerGap())
        );

        hubTab.addTab("Neighbor View", neighborPanel);

        javax.swing.GroupLayout selectedImagePanelClassLayout = new javax.swing.GroupLayout(selectedImagePanelClass);
        selectedImagePanelClass.setLayout(selectedImagePanelClassLayout);
        selectedImagePanelClassLayout.setHorizontalGroup(
            selectedImagePanelClassLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 222, Short.MAX_VALUE)
        );
        selectedImagePanelClassLayout.setVerticalGroup(
            selectedImagePanelClassLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 224, Short.MAX_VALUE)
        );

        selectedImageLabelClass.setText("Current Image");

        confusionMatScrollPane.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        confusionMatScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        classHubnessTable.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                {null, null, null, null},
                {null, null, null, null},
                {null, null, null, null},
                {null, null, null, null}
            },
            new String [] {
                "Title 1", "Title 2", "Title 3", "Title 4"
            }
        ));
        classHubnessTable.setAutoResizeMode(javax.swing.JTable.AUTO_RESIZE_OFF);
        classHubnessTable.setMaximumSize(new java.awt.Dimension(2000, 2000));
        classHubnessTable.setMinimumSize(new java.awt.Dimension(120, 120));
        classHubnessTable.setRowHeight(30);
        confusionMatScrollPane.setViewportView(classHubnessTable);

        classesScrollPane.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        classesScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        classesScrollPanel.setPreferredSize(new java.awt.Dimension(760, 1508));

        javax.swing.GroupLayout classesScrollPanelLayout = new javax.swing.GroupLayout(classesScrollPanel);
        classesScrollPanel.setLayout(classesScrollPanelLayout);
        classesScrollPanelLayout.setHorizontalGroup(
            classesScrollPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 760, Short.MAX_VALUE)
        );
        classesScrollPanelLayout.setVerticalGroup(
            classesScrollPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 1508, Short.MAX_VALUE)
        );

        classesScrollPane.setViewportView(classesScrollPanel);

        javax.swing.GroupLayout classDistributionHolderLayout = new javax.swing.GroupLayout(classDistributionHolder);
        classDistributionHolder.setLayout(classDistributionHolderLayout);
        classDistributionHolderLayout.setHorizontalGroup(
            classDistributionHolderLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 428, Short.MAX_VALUE)
        );
        classDistributionHolderLayout.setVerticalGroup(
            classDistributionHolderLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );

        selectedImagePathLabelClass.setText("Path:");
        jScrollPane3.setViewportView(selectedImagePathLabelClass);

        cNamesScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
        cNamesScrollPane.setMaximumSize(new java.awt.Dimension(100, 100));
        cNamesScrollPane.setMinimumSize(new java.awt.Dimension(100, 100));

        javax.swing.GroupLayout classColorAndNamesPanelLayout = new javax.swing.GroupLayout(classColorAndNamesPanel);
        classColorAndNamesPanel.setLayout(classColorAndNamesPanelLayout);
        classColorAndNamesPanelLayout.setHorizontalGroup(
            classColorAndNamesPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 350, Short.MAX_VALUE)
        );
        classColorAndNamesPanelLayout.setVerticalGroup(
            classColorAndNamesPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 255, Short.MAX_VALUE)
        );

        jScrollPane5.setViewportView(classColorAndNamesPanel);

        cNamesScrollPane.setViewportView(jScrollPane5);

        javax.swing.GroupLayout classPanelLayout = new javax.swing.GroupLayout(classPanel);
        classPanel.setLayout(classPanelLayout);
        classPanelLayout.setHorizontalGroup(
            classPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(classPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(classPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(classesScrollPane, javax.swing.GroupLayout.DEFAULT_SIZE, 764, Short.MAX_VALUE)
                    .addGroup(classPanelLayout.createSequentialGroup()
                        .addComponent(classDistributionHolder, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(18, 18, 18)
                        .addComponent(cNamesScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(classPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(classPanelLayout.createSequentialGroup()
                        .addGroup(classPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addGroup(classPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                .addComponent(selectedImageLabelClass, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(selectedImagePanelClass, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                            .addComponent(confusionMatScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 262, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(21, 21, 21))
                    .addGroup(classPanelLayout.createSequentialGroup()
                        .addComponent(jScrollPane3, javax.swing.GroupLayout.PREFERRED_SIZE, 278, Short.MAX_VALUE)
                        .addContainerGap())))
        );
        classPanelLayout.setVerticalGroup(
            classPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(classPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(selectedImagePanelClass, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(selectedImageLabelClass, javax.swing.GroupLayout.PREFERRED_SIZE, 27, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane3, javax.swing.GroupLayout.DEFAULT_SIZE, 86, Short.MAX_VALUE)
                .addGap(18, 18, 18)
                .addComponent(confusionMatScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 237, javax.swing.GroupLayout.PREFERRED_SIZE))
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, classPanelLayout.createSequentialGroup()
                .addGroup(classPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(classDistributionHolder, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(classPanelLayout.createSequentialGroup()
                        .addComponent(cNamesScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 244, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(0, 32, Short.MAX_VALUE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(classesScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 331, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(11, 11, 11))
        );

        hubTab.addTab("Class View", classPanel);

        javax.swing.GroupLayout selectedImagePanelSearchLayout = new javax.swing.GroupLayout(selectedImagePanelSearch);
        selectedImagePanelSearch.setLayout(selectedImagePanelSearchLayout);
        selectedImagePanelSearchLayout.setHorizontalGroup(
            selectedImagePanelSearchLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 222, Short.MAX_VALUE)
        );
        selectedImagePanelSearchLayout.setVerticalGroup(
            selectedImagePanelSearchLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 224, Short.MAX_VALUE)
        );

        selectedImageLabelSearch.setText("Current Image");

        searchQLabelTxt.setText("Do you want to search the image collection?");

        javax.swing.GroupLayout queryImagePanelLayout = new javax.swing.GroupLayout(queryImagePanel);
        queryImagePanel.setLayout(queryImagePanelLayout);
        queryImagePanelLayout.setHorizontalGroup(
            queryImagePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 237, Short.MAX_VALUE)
        );
        queryImagePanelLayout.setVerticalGroup(
            queryImagePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 237, Short.MAX_VALUE)
        );

        imageBrowseButton.setText("Browse");
        imageBrowseButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                imageBrowseButtonActionPerformed(evt);
            }
        });

        jTextField1.setText("                       -- Enter text --");
        jTextField1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jTextField1ActionPerformed(evt);
            }
        });

        queryQTextLabelTxt.setText("You can also search with a textual query:");

        queryNNScrollPane.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        queryNNScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        javax.swing.GroupLayout queryNNPanelLayout = new javax.swing.GroupLayout(queryNNPanel);
        queryNNPanel.setLayout(queryNNPanelLayout);
        queryNNPanelLayout.setHorizontalGroup(
            queryNNPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 184, Short.MAX_VALUE)
        );
        queryNNPanelLayout.setVerticalGroup(
            queryNNPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 432, Short.MAX_VALUE)
        );

        queryNNScrollPane.setViewportView(queryNNPanel);

        simResLabelTxt.setText("Here is a list of most similar results:");

        searchButton.setText("SEARCH");
        searchButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                searchButtonActionPerformed(evt);
            }
        });

        prClassLabelTxt.setText("Predicted class for k = 10:");

        prClassScrollPane.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        prClassScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        javax.swing.GroupLayout classifierPredictionsPanelLayout = new javax.swing.GroupLayout(classifierPredictionsPanel);
        classifierPredictionsPanel.setLayout(classifierPredictionsPanelLayout);
        classifierPredictionsPanelLayout.setHorizontalGroup(
            classifierPredictionsPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 241, Short.MAX_VALUE)
        );
        classifierPredictionsPanelLayout.setVerticalGroup(
            classifierPredictionsPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 432, Short.MAX_VALUE)
        );

        prClassScrollPane.setViewportView(classifierPredictionsPanel);

        collectionSearchButton.setText("Select from collection");
        collectionSearchButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                collectionSearchButtonActionPerformed(evt);
            }
        });

        selectedImagePathLabelSearch.setText("Path:");
        jScrollPane4.setViewportView(selectedImagePathLabelSearch);

        reRankingButton.setText("Re-rank");
        reRankingButton.setToolTipText("Perform Secondary\nHubness-aware Re-ranking");
        reRankingButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                reRankingButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout searchPanelLayout = new javax.swing.GroupLayout(searchPanel);
        searchPanel.setLayout(searchPanelLayout);
        searchPanelLayout.setHorizontalGroup(
            searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(searchPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(searchQLabelTxt, javax.swing.GroupLayout.PREFERRED_SIZE, 319, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                        .addComponent(queryQTextLabelTxt, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(jTextField1, javax.swing.GroupLayout.Alignment.LEADING)
                        .addComponent(searchButton, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                        .addGroup(javax.swing.GroupLayout.Alignment.LEADING, searchPanelLayout.createSequentialGroup()
                            .addComponent(imageBrowseButton)
                            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                            .addComponent(collectionSearchButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addComponent(queryImagePanel, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(searchPanelLayout.createSequentialGroup()
                        .addGap(18, 18, 18)
                        .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(simResLabelTxt, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(queryNNScrollPane))
                        .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(searchPanelLayout.createSequentialGroup()
                                .addGap(44, 44, 44)
                                .addComponent(prClassLabelTxt, javax.swing.GroupLayout.PREFERRED_SIZE, 226, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(searchPanelLayout.createSequentialGroup()
                                .addGap(29, 29, 29)
                                .addComponent(prClassScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 260, javax.swing.GroupLayout.PREFERRED_SIZE))))
                    .addGroup(searchPanelLayout.createSequentialGroup()
                        .addGap(59, 59, 59)
                        .addComponent(reRankingButton, javax.swing.GroupLayout.PREFERRED_SIZE, 119, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane4, javax.swing.GroupLayout.PREFERRED_SIZE, 236, Short.MAX_VALUE)
                    .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                        .addComponent(selectedImageLabelSearch, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(selectedImagePanelSearch, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addContainerGap())
        );
        searchPanelLayout.setVerticalGroup(
            searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(searchPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(searchPanelLayout.createSequentialGroup()
                        .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(searchPanelLayout.createSequentialGroup()
                                .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(searchQLabelTxt, javax.swing.GroupLayout.PREFERRED_SIZE, 30, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(simResLabelTxt, javax.swing.GroupLayout.PREFERRED_SIZE, 51, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(prClassLabelTxt))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                    .addComponent(imageBrowseButton)
                                    .addComponent(collectionSearchButton)
                                    .addComponent(reRankingButton))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(queryImagePanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(searchPanelLayout.createSequentialGroup()
                                .addComponent(selectedImagePanelSearch, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(selectedImageLabelSearch, javax.swing.GroupLayout.PREFERRED_SIZE, 27, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(jScrollPane4, javax.swing.GroupLayout.PREFERRED_SIZE, 69, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addGap(13, 13, 13)
                        .addComponent(queryQTextLabelTxt)
                        .addGap(18, 18, 18)
                        .addComponent(jTextField1, javax.swing.GroupLayout.PREFERRED_SIZE, 103, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(searchButton, javax.swing.GroupLayout.DEFAULT_SIZE, 111, Short.MAX_VALUE))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, searchPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                        .addComponent(prClassScrollPane)
                        .addComponent(queryNNScrollPane)))
                .addContainerGap())
        );

        hubTab.addTab("Search", searchPanel);

        collectionMenu.setLabel("Collection");

        workspaceMenuItem.setText("Select workspace");
        workspaceMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                workspaceMenuItemActionPerformed(evt);
            }
        });
        collectionMenu.add(workspaceMenuItem);

        importItem.setLabel("Import data");
        importItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                importItemActionPerformed(evt);
            }
        });
        collectionMenu.add(importItem);

        dMatrixMenu.setText("Distances and Neighbor Sets");

        distImportItem.setText("Import");
        distImportItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                distImportItemActionPerformed(evt);
            }
        });
        dMatrixMenu.add(distImportItem);

        distCalculateMenu.setText("Calculate");

        manhattanDistItem.setText("Manhattan");
        manhattanDistItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                manhattanDistItemActionPerformed(evt);
            }
        });
        distCalculateMenu.add(manhattanDistItem);

        distCalcEuclideanItem.setText("Euclidean");
        distCalcEuclideanItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                distCalcEuclideanItemActionPerformed(evt);
            }
        });
        distCalculateMenu.add(distCalcEuclideanItem);

        distCalcCosineItem.setText("Cosine");
        distCalcCosineItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                distCalcCosineItemActionPerformed(evt);
            }
        });
        distCalculateMenu.add(distCalcCosineItem);

        tanimotoMenuItem.setText("Tanimoto");
        tanimotoMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                tanimotoMenuItemActionPerformed(evt);
            }
        });
        distCalculateMenu.add(tanimotoMenuItem);

        klMenuItem.setText("KL divergence");
        klMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                klMenuItemActionPerformed(evt);
            }
        });
        distCalculateMenu.add(klMenuItem);

        bcMenuItem.setText("Bray-Curtis");
        bcMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                bcMenuItemActionPerformed(evt);
            }
        });
        distCalculateMenu.add(bcMenuItem);

        canMenuItem.setText("Canberra");
        canMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                canMenuItemActionPerformed(evt);
            }
        });
        distCalculateMenu.add(canMenuItem);

        dMatrixMenu.add(distCalculateMenu);

        collectionMenu.add(dMatrixMenu);

        neighborStatsItem.setText("Calculate Neighbor Stats");
        neighborStatsItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                neighborStatsItemActionPerformed(evt);
            }
        });
        collectionMenu.add(neighborStatsItem);

        mdsVisualizeItem.setText("MDS Visualize");
        mdsVisualizeItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                mdsVisualizeItemActionPerformed(evt);
            }
        });
        collectionMenu.add(mdsVisualizeItem);

        selImgPathMenuItem.setText("Select image by browsing");
        selImgPathMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                selImgPathMenuItemActionPerformed(evt);
            }
        });
        collectionMenu.add(selImgPathMenuItem);

        majorHubSelectionItem.setText("Select major hub");
        majorHubSelectionItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                majorHubSelectionItemActionPerformed(evt);
            }
        });
        collectionMenu.add(majorHubSelectionItem);

        menuBar.add(collectionMenu);

        metricLearningMenu.setText("Metric Learning");

        secondaryMetricMenu.setText("Calculate secondary metric");

        simcosMenuItem.setText("simcos shared neighbor sim");
        simcosMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                simcosMenuItemActionPerformed(evt);
            }
        });
        secondaryMetricMenu.add(simcosMenuItem);

        simhubMenuItem.setText("simhub shared neighbor sim");
        simhubMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                simhubMenuItemActionPerformed(evt);
            }
        });
        secondaryMetricMenu.add(simhubMenuItem);

        mpMenuItem.setText("mutual proximity");
        mpMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                mpMenuItemActionPerformed(evt);
            }
        });
        secondaryMetricMenu.add(mpMenuItem);

        localScalingItem.setText("local scaling");
        localScalingItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                localScalingItemActionPerformed(evt);
            }
        });
        secondaryMetricMenu.add(localScalingItem);

        nicdmItem.setText("NICDM");
        nicdmItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                nicdmItemActionPerformed(evt);
            }
        });
        secondaryMetricMenu.add(nicdmItem);

        metricLearningMenu.add(secondaryMetricMenu);

        loadSecondaryDistancesItem.setText("Load secondary distance matrix");
        loadSecondaryDistancesItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadSecondaryDistancesItemActionPerformed(evt);
            }
        });
        metricLearningMenu.add(loadSecondaryDistancesItem);

        menuBar.add(metricLearningMenu);

        editMenu.setText("Edit");

        previousMenuItem.setText("Previous");
        previousMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                previousMenuItemActionPerformed(evt);
            }
        });
        editMenu.add(previousMenuItem);

        nextMenuItem.setText("Next");
        nextMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                nextMenuItemActionPerformed(evt);
            }
        });
        editMenu.add(nextMenuItem);

        screenCaptureMenu.setText("Screen capture");

        mdsScreenCaptureItem.setText("MDS screen");
        mdsScreenCaptureItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                mdsScreenCaptureItemActionPerformed(evt);
            }
        });
        screenCaptureMenu.add(mdsScreenCaptureItem);

        graphScreenCaptureItem.setText("Graph screen");
        graphScreenCaptureItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                graphScreenCaptureItemActionPerformed(evt);
            }
        });
        screenCaptureMenu.add(graphScreenCaptureItem);

        editMenu.add(screenCaptureMenu);

        menuBar.add(editMenu);

        codebookMenu.setText("Codebook");

        loadCodebookItem.setText("load Codebook");
        loadCodebookItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadCodebookItemActionPerformed(evt);
            }
        });
        codebookMenu.add(loadCodebookItem);

        loadCodebookProfileMenuItem.setText("load Codebook Profile");
        loadCodebookProfileMenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                loadCodebookProfileMenuItemActionPerformed(evt);
            }
        });
        codebookMenu.add(loadCodebookProfileMenuItem);

        menuBar.add(codebookMenu);

        classificationMenu.setText("Classification");

        trainModelsItem.setText("Train models");
        trainModelsItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                trainModelsItemActionPerformed(evt);
            }
        });
        classificationMenu.add(trainModelsItem);

        menuBar.add(classificationMenu);

        selImageMenu.setText("Selected Image");

        selSIFTmenuItem.setText("Visual words assessment view");
        selSIFTmenuItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                selSIFTmenuItemActionPerformed(evt);
            }
        });
        selImageMenu.add(selSIFTmenuItem);

        menuBar.add(selImageMenu);

        setJMenuBar(menuBar);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(hubTab, javax.swing.GroupLayout.PREFERRED_SIZE, 1073, Short.MAX_VALUE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(hubTab)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

private void mdsCollectionPanelMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_mdsCollectionPanelMouseClicked
}//GEN-LAST:event_mdsCollectionPanelMouseClicked

private void jTextField1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jTextField1ActionPerformed
}//GEN-LAST:event_jTextField1ActionPerformed

    /**
     * This method sets the workspace to the selected directory.
     *
     * @param evt ActionEvent object.
     */
private void workspaceMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_workspaceMenuItemActionPerformed
    JFileChooser jfc = new JFileChooser(currentDirectory);
    jfc.setDialogTitle("Set Workspace");
    jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
    int rVal = jfc.showOpenDialog(ImageHubExplorer.this);
    if (rVal == JFileChooser.APPROVE_OPTION) {
        currentDirectory = jfc.getSelectedFile();
        workspace = jfc.getSelectedFile();
        workspaceLabelValue.setText(workspace.getAbsolutePath());
    }
}//GEN-LAST:event_workspaceMenuItemActionPerformed

    /**
     * This class is used for calculating the primary distance matrix.
     */
    private class DistanceCalcHelperPrimary implements Runnable {

        boolean calcNSets = true;
        File dir;

        /**
         * Initialization.
         *
         * @param calcNSets Boolean flag indicating whether to also calculate
         * the kNN sets at once.
         * @param dir Directory where to persist the matrix and the kNN sets.
         */
        public DistanceCalcHelperPrimary(boolean calcNSets, File dir) {
            this.calcNSets = calcNSets;
            this.dir = dir;
        }

        @Override
        public void run() {
            try {
                distMatrixPrimary = quantizedRepresentation.
                        calculateDistMatrixMultThr(primaryCMet, 8);
                JOptionPane.showMessageDialog(frameReference,
                        "Distances properly calculated.");
                if (calcNSets) {
                    Thread t = new Thread(new NSetCalcHelperPrimary());
                    t.start();
                    try {
                        t.join();
                    } catch (Throwable thr) {
                        System.err.println(thr.getMessage());
                    }
                    nsfPrimary.setDistances(distMatrixPrimary);
                }
                printDMatToFile(distMatrixPrimary, new File(dir, "dMat.txt"));
                nsfPrimary.saveNeighborSets(new File(dir, "knnSets.txt"));
                imageCoordinatesXY = null;
            } catch (Exception e) {
                JOptionPane.showMessageDialog(frameReference,
                        "An error occurred: " + e.getMessage(),
                        "Error message", JOptionPane.ERROR_MESSAGE);
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * This class is used for calculating the kNN sets in the primary metric.
     */
    private class NSetCalcHelperPrimary implements Runnable {

        /**
         * The default constructor.
         */
        public NSetCalcHelperPrimary() {
        }

        @Override
        public void run() {
            try {
                nsfPrimary = new NeighborSetFinder(
                        quantizedRepresentation, distMatrixPrimary,
                        primaryCMet);
                nsfPrimary.calculateNeighborSetsMultiThr(50, 8);
            } catch (Exception e) {
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * This method calculates the Manhattan distances from the representation.
     *
     * @param evt ActionEvent object.
     */
private void manhattanDistItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_manhattanDistItemActionPerformed
    if (!busyCalculating) {
        busyCalculating = true;
        try {
            primaryCMet = CombinedMetric.FLOAT_MANHATTAN;
            // Designate a file to save the matrix to.
            primaryDMatFile = new File(workspace, "distancesNNSets"
                    + File.separator
                    + primaryCMet.getFloatMetric().getClass().getName()
                    + File.separator + "dMat.txt");
            boolean calcNSets = !(new File(primaryDMatFile.getParentFile(),
                    "knnSets.txt")).exists();
            if (quantizedRepresentation != null
                    && !quantizedRepresentation.isEmpty()
                    && !primaryDMatFile.exists()) {
                Thread t = new Thread(new DistanceCalcHelperPrimary(
                        calcNSets, primaryDMatFile.getParentFile()));
                t.start();
            } else if (primaryDMatFile.exists()) {
                // If these distances already exist, just load them.
                loadDistancesAndNeighbors(primaryDMatFile, PRIMARY_METRIC);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            busyCalculating = false;
        }
    }
}//GEN-LAST:event_manhattanDistItemActionPerformed

    /**
     * This method calculates the Euclidean distances from the representation.
     *
     * @param evt ActionEvent object.
     */
private void distCalcEuclideanItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_distCalcEuclideanItemActionPerformed
    if (!busyCalculating) {
        busyCalculating = true;
        try {
            primaryCMet = CombinedMetric.FLOAT_EUCLIDEAN;
            // Designate a file to save the matrix to.
            primaryDMatFile = new File(workspace,
                    "distancesNNSets" + File.separator
                    + primaryCMet.getFloatMetric().getClass().getName()
                    + File.separator + "dMat.txt");
            boolean calcNSets = !(new File(primaryDMatFile.getParentFile(),
                    "knnSets.txt")).exists();
            if (quantizedRepresentation != null
                    && !quantizedRepresentation.isEmpty()
                    && !primaryDMatFile.exists()) {
                Thread t = new Thread(new DistanceCalcHelperPrimary(
                        calcNSets, primaryDMatFile.getParentFile()));
                t.start();
            } else if (primaryDMatFile.exists()) {
                // If these distances already exist, just load them.
                loadDistancesAndNeighbors(primaryDMatFile, PRIMARY_METRIC);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            busyCalculating = false;
        }
    }
}//GEN-LAST:event_distCalcEuclideanItemActionPerformed

    /**
     * This method calculates the cosine distances from the representation.
     *
     * @param evt ActionEvent object.
     */
private void distCalcCosineItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_distCalcCosineItemActionPerformed
    if (!busyCalculating) {
        busyCalculating = true;
        try {
            primaryCMet = CombinedMetric.FLOAT_COSINE;
            // Designate a file to save the matrix to.
            primaryDMatFile = new File(workspace, "distancesNNSets"
                    + File.separator
                    + primaryCMet.getFloatMetric().getClass().getName()
                    + File.separator + "dMat.txt");
            boolean calcNSets = !(new File(primaryDMatFile.getParentFile(),
                    "knnSets.txt")).exists();
            if (quantizedRepresentation != null
                    && !quantizedRepresentation.isEmpty()
                    && !primaryDMatFile.exists()) {
                Thread t = new Thread(new DistanceCalcHelperPrimary(calcNSets,
                        primaryDMatFile.getParentFile()));
                t.start();
            } else if (primaryDMatFile.exists()) {
                // If these distances already exist, just load them.
                loadDistancesAndNeighbors(primaryDMatFile, PRIMARY_METRIC);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            busyCalculating = false;
        }
    }
}//GEN-LAST:event_distCalcCosineItemActionPerformed

    /**
     * This method imports the distances and neighbor sets from the specified
     * files.
     *
     * @param evt ActionEvent object.
     */
private void distImportItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_distImportItemActionPerformed
    JFileChooser jfc = new JFileChooser(currentDirectory);
    jfc.setDialogTitle("Choose distances file. "
            + "Neighbor sets will be automatically loaded if available");
    jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
    int rVal = jfc.showOpenDialog(ImageHubExplorer.this);
    if (rVal == JFileChooser.APPROVE_OPTION) {
        currentDirectory = jfc.getSelectedFile().getParentFile();
        primaryDMatFile = jfc.getSelectedFile();
        busyCalculating = true;
        try {
            loadDistancesAndNeighbors(primaryDMatFile, PRIMARY_METRIC);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            busyCalculating = false;
        }
    }
}//GEN-LAST:event_distImportItemActionPerformed

    /**
     * This method performs the data representation import.
     *
     * @param evt ActionEvent object.
     */
private void importItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_importItemActionPerformed
    Thread t = new Thread(new ImportHelper());
    t.start();
}//GEN-LAST:event_importItemActionPerformed

    /**
     * This method loads the codebook definition from the specified file.
     *
     * @param evt ActionEvent object.
     */
private void loadCodebookItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadCodebookItemActionPerformed
    JFileChooser jfc = new JFileChooser(currentDirectory);
    jfc.setDialogTitle("Load SIFT codebook");
    jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
    int rVal = jfc.showOpenDialog(ImageHubExplorer.this);
    if (rVal == JFileChooser.APPROVE_OPTION) {
        currentDirectory = jfc.getSelectedFile().getParentFile();
        File cbFile = jfc.getSelectedFile();
        busyCalculating = true;
        try {
            codebook = new GenericCodeBook();
            codebook.loadCodeBookFromFile(cbFile);
            JOptionPane.showMessageDialog(frameReference, "Load completed");
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            busyCalculating = false;
        }
    }
}//GEN-LAST:event_loadCodebookItemActionPerformed

    /**
     * This method selects and image as the current image by browsing through
     * the image dataset directly and selecting the corresponding image file.
     *
     * @param evt ActionEvent object.
     */
private void selImgPathMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_selImgPathMenuItemActionPerformed
    JFileChooser jfc = new JFileChooser(currentDirectory);
    jfc.setDialogTitle("Select an image");
    jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
    int rVal = jfc.showOpenDialog(ImageHubExplorer.this);
    if (rVal == JFileChooser.APPROVE_OPTION) {
        currentDirectory = jfc.getSelectedFile().getParentFile();
        File imageFile = jfc.getSelectedFile();
        busyCalculating = true;
        try {
            // Look up the image index in the dataset by retrieving it from
            // the image path map.
            int index = pathIndexMap.get(imageFile.getPath());
            setSelectedImageForIndex(index);
            JOptionPane.showMessageDialog(frameReference,
                    "Selection performed");
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            busyCalculating = false;
        }
    }
}//GEN-LAST:event_selImgPathMenuItemActionPerformed

    /**
     * This method shifts the focus back to the previously examined image.
     *
     * @param evt ActionEvent object.
     */
private void previousMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_previousMenuItemActionPerformed
    if (selectedImageIndexInHistory > 0) {
        selectedImageIndexInHistory--;
    }
    setSelectedImageForHistoryIndex(selectedImageIndexInHistory);
}//GEN-LAST:event_previousMenuItemActionPerformed

    /**
     * This method shifts the focus forward to the next image in the history.
     *
     * @param evt ActionEvent object.
     */
private void nextMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_nextMenuItemActionPerformed
    if (selectedImageIndexInHistory < selectedImageHistory.size() - 1) {
        selectedImageIndexInHistory++;
    }
    setSelectedImageForHistoryIndex(selectedImageIndexInHistory);
}//GEN-LAST:event_nextMenuItemActionPerformed

    /**
     * This method loads the secondary distances and neighbor sets from the
     * specified files, if available.
     *
     * @param evt ActionEvent object.
     */
private void loadSecondaryDistancesItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadSecondaryDistancesItemActionPerformed
    JFileChooser jfc = new JFileChooser(currentDirectory);
    jfc.setDialogTitle("Choose distances file. Neighbor sets will be "
            + "automatically loaded if available");
    jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
    int rVal = jfc.showOpenDialog(ImageHubExplorer.this);
    if (rVal == JFileChooser.APPROVE_OPTION) {
        currentDirectory = jfc.getSelectedFile().getParentFile();
        secondaryDMatFile = jfc.getSelectedFile();
        busyCalculating = true;
        try {
            loadDistancesAndNeighbors(secondaryDMatFile, SECONDARY_METRIC);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            busyCalculating = false;
        }
    }
}//GEN-LAST:event_loadSecondaryDistancesItemActionPerformed

    /**
     * Calculate the secondary simcos distances.
     *
     * @param evt ActionEvent object.
     */
private void simcosMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_simcosMenuItemActionPerformed
    Thread t = new Thread(new SimCosHelper());
    t.start();
}//GEN-LAST:event_simcosMenuItemActionPerformed

    private class SimCosHelper implements Runnable {

        /**
         * The default constructor.
         */
        public SimCosHelper() {
        }

        @Override
        public void run() {
            if (busyCalculating) {
                // If the system is currently performing calculations, abort the
                // call.
                return;
            }
            busyCalculating = true;
            try {
                // The primary and secondary matrix files.
                File pDir = primaryDMatFile.getParentFile();
                File sDir = new File(pDir, "secondary" + File.separator
                        + "simcos");
                File matFile = new File(sDir, "dMat.txt");
                if (matFile.exists() && secondaryLoadFlag) {
                    // Load if it exists.
                    distMatrixSecondary = loadDMatFromFile(matFile);
                } else {
                    // Calculate the distances if it doesn't.
                    SharedNeighborFinder snf =
                            new SharedNeighborFinder(nsfPrimary);
                    snf.setNumClasses(numClasses);
                    snf.setPrimaryMetricsCalculator(primaryCMet);
                    snf.countSharedNeighborsMultiThread(8);
                    // What we first get is the similarities, so we'll need to
                    // subtract them from the max to get the distances.
                    distMatrixSecondary = snf.getSharedNeighborCounts();
                    for (int i = 0; i < distMatrixSecondary.length; i++) {
                        for (int j = 0; j < distMatrixSecondary[i].length;
                                j++) {
                            distMatrixSecondary[i][j] =
                                    50 - distMatrixSecondary[i][j];
                        }
                    }
                    // Set the secondary CombinedMetric object.
                    secondaryCMet = new SharedNeighborCalculator(
                            snf, SharedNeighborCalculator.WeightingType.NONE);
                }
                // Load or calculate the secondary kNN sets.
                File neighborsFile = new File(sDir, "knnSets.txt");
                if (neighborsFile.exists() && secondaryLoadFlag) {
                    nsfSecondary = NeighborSetFinder.loadNSF(neighborsFile,
                            quantizedRepresentation);
                    nsfSecondary.setDistances(distMatrixSecondary);
                } else {
                    nsfSecondary = new NeighborSetFinder(
                            quantizedRepresentation, distMatrixSecondary,
                            secondaryCMet);
                    nsfSecondary.calculateNeighborSetsMultiThr(50, 8);
                }
                printDMatToFile(distMatrixSecondary, matFile);
                nsfSecondary.saveNeighborSets(neighborsFile);
                neighborStatsCalculated = false;
                imageCoordinatesXY = null;
                JOptionPane.showMessageDialog(frameReference,
                        "Simcos calculated");
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * Calculate the simhub secondary distances.
     *
     * @param evt ActionEvent object.
     */
private void simhubMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_simhubMenuItemActionPerformed
    Thread t = new Thread(new SimHubHelper());
    t.start();
}//GEN-LAST:event_simhubMenuItemActionPerformed

    /**
     * Calculate the secondary mutual proximity distances.
     *
     * @param evt ActionEvent object.
     */
private void mpMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_mpMenuItemActionPerformed
    Thread t = new Thread(new MutualProximityHelper());
    t.start();
}//GEN-LAST:event_mpMenuItemActionPerformed

    /**
     * Calculate the secondary local scaling distances.
     *
     * @param evt ActionEvent object.
     */
private void localScalingItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_localScalingItemActionPerformed
    Thread t = new Thread(new LocalScalingHelper());
    t.start();
}//GEN-LAST:event_localScalingItemActionPerformed

    /**
     * Calculate the secondary nicdm features.
     *
     * @param evt ActionEvent object.
     */
private void nicdmItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_nicdmItemActionPerformed
    Thread t = new Thread(new NICDMHelper());
    t.start();
}//GEN-LAST:event_nicdmItemActionPerformed

    /**
     * Calculate the stats of the kNN sets and the neighbor occurrence
     * distribution.
     *
     * @param evt ActionEvent object.
     */
private void neighborStatsItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_neighborStatsItemActionPerformed
    Thread t = new Thread(new NSFStatsHelper());
    t.start();
}//GEN-LAST:event_neighborStatsItemActionPerformed

    /**
     * This method selects the major image hub as the currently selected image.
     *
     * @param evt ActionEvent object.
     */
private void majorHubSelectionItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_majorHubSelectionItemActionPerformed
    try {
        NeighborSetFinder nsf = getNSF().copy();
        nsf.recalculateStatsForSmallerK(neighborhoodSize);
        int[] hubness = nsf.getNeighborFrequencies();
        int[] maxWithIndex = ArrayUtil.maxWithIndex(hubness);
        int index = maxWithIndex[1];
        setSelectedImageForIndex(index);
        System.out.println("Selected hub index " + index);
    } catch (Exception e) {
        System.err.println(e.getMessage());
    }
}//GEN-LAST:event_majorHubSelectionItemActionPerformed

    /**
     * Queries the data by an image from the represented collection.
     *
     * @param evt ActionEvent object.
     */
private void collectionSearchButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_collectionSearchButtonActionPerformed
    if (busyCalculating) {
        // If the system is working on something else, abort.
        return;
    }
    try {
        setQueryImageFromCollection();
    } catch (Exception e) {
        System.err.println(e.getMessage());
    }
}//GEN-LAST:event_collectionSearchButtonActionPerformed

    /**
     * Select an image for the image query and extract its features.
     *
     * @param evt ActionEvent object.
     */
private void imageBrowseButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_imageBrowseButtonActionPerformed
    JFileChooser jfc = new JFileChooser(currentDirectory);
    jfc.setDialogTitle("Select Image Query");
    jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
    int rVal = jfc.showOpenDialog(ImageHubExplorer.this);
    if (rVal == JFileChooser.APPROVE_OPTION) {
        currentDirectory = jfc.getSelectedFile().getParentFile();
        try {
            File imageFile = jfc.getSelectedFile();
            queryImage = ImageIO.read(imageFile);
            queryImagePanel.setImage(queryImage);
            queryImagePanel.revalidate();
            queryImagePanel.repaint();
            if (codebook == null) {
                // If the codebook has not been loaded, it is impossible to
                // gemerate the quantized representation.
                JOptionPane.showMessageDialog(frameReference,
                        "First load the codebook", "Error message",
                        JOptionPane.ERROR_MESSAGE);
                return;
            }
            String imgName = imageFile.getName();
            int pointIndex = imgName.lastIndexOf('.');
            String shortName = null;
            if (pointIndex != -1) {
                shortName = imgName.substring(0, pointIndex);
                shortName += ".pgm";
            } else {
                shortName = shortName + ".pgm";
            }
            // Generate the PGM file for SiftWin SIFT feature extraction.
            File pgmFile = new File(workspace, "tmp" + File.separator
                    + shortName);
            File siftTempFile = new File(workspace, "tmp" + File.separator
                    + "siftTemp.key");
            ConvertJPGToPGM.convertFile(imageFile, pgmFile);
            SiftUtil.siftFile(pgmFile, siftTempFile, "");
            pgmFile.delete();
            // Load the extracted SIFT features.
            queryImageLFeat = SiftUtil.importFeaturesFromSift(siftTempFile);
            DataInstance queryImageRepAlmost =
                    codebook.getDistributionForImageRepresentation(
                    queryImageLFeat, imageFile.getPath());
            // Get the color histogram.
            ColorHistogramVector cHist = new ColorHistogramVector();
            cHist.populateFromImage(queryImage, imgName);
            queryImageRep = new DataInstance();
            queryImageRep.fAttr = new float[queryImageRepAlmost.getNumFAtt()
                    + cHist.getNumFAtt()];
            queryImageRep.iAttr = new int[queryImageRepAlmost.getNumIAtt()];
            queryImageRep.sAttr = new String[queryImageRepAlmost.getNumNAtt()];
            System.arraycopy(cHist.fAttr, 0, queryImageRep.fAttr, 0,
                    cHist.getNumFAtt());
            for (int i = 0; i < queryImageRepAlmost.getNumFAtt(); i++) {
                queryImageRep.fAttr[i + cHist.getNumFAtt()] =
                        queryImageRepAlmost.fAttr[i];
                if (queryImageRep.iAttr != null
                        && queryImageRepAlmost.iAttr != null) {
                    queryImageRep.iAttr[i] = queryImageRepAlmost.iAttr[i];
                }
                if (queryImageRep.sAttr != null
                        && queryImageRepAlmost.sAttr != null) {
                    queryImageRep.sAttr[i] = queryImageRepAlmost.sAttr[i];
                }
            }
            // Reset the neighbor lists and the predictions in the query panel.
            queryNNPanel.removeAll();
            queryNNPanel.revalidate();
            queryNNPanel.repaint();

            classifierPredictionsPanel.removeAll();
            classifierPredictionsPanel.revalidate();
            classifierPredictionsPanel.repaint();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
}//GEN-LAST:event_imageBrowseButtonActionPerformed

    /**
     * Run the image query against the represented image dataset.
     *
     * @param evt ActionEvent object.
     */
private void searchButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_searchButtonActionPerformed
    if (queryImageRep != null && queryImage != null) {
        imageQuery();
    }
}//GEN-LAST:event_searchButtonActionPerformed

    /**
     * This method trains the classifier models on the represented image data.
     *
     * @param evt ActionEvent object.
     */
private void trainModelsItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_trainModelsItemActionPerformed
    trainModels();
}//GEN-LAST:event_trainModelsItemActionPerformed

    /**
     * Perform re-ranking on the query result set.
     *
     * @param evt ActionEvent object.
     */
private void reRankingButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_reRankingButtonActionPerformed
    if (busyCalculating || queryImage == null
            || queryImageNeighborDists == null) {
        return;
    }
    busyCalculating = true;
    try {
        // Get the neighbor occurrence profiles for the k of the query.
        NeighborSetFinder nsf = getNSF().copy();
        nsf.recalculateStatsForSmallerK(kQuery);
        int[] hubness = nsf.getNeighborFrequencies();
        int[] badHubness = nsf.getBadFrequencies();
        // Re-scale the distances to the neighbors.
        for (int i = 0; i < queryImageNeighborDists.length; i++) {
            if (hubness[queryImageNeighbors[i]] > 0) {
                queryImageNeighborDists[i] *= ((float) (
                        badHubness[queryImageNeighbors[i]]))
                        / ((float) (hubness[queryImageNeighbors[i]]));
            }
        }
        // Sort according to the new distances.
        int[] permutationIndexes = AuxSort.sortIndexedValue(
                queryImageNeighborDists, false);
        // Now permute the query neighbors.
        int[] qNeighbors = new int[queryImageNeighbors.length];
        for (int i = 0; i < queryImageNeighborDists.length; i++) {
            qNeighbors[i] = queryImageNeighbors[permutationIndexes[i]];
        }
        // Empty the query neighbors display.
        queryImageNeighbors = qNeighbors;
        queryNNPanel.removeAll();
        queryNNPanel.revalidate();
        queryNNPanel.repaint();
        // Insert elements into the query neighbor display according to the
        // re-ranked kNN set.
        for (int i = 0; i < queryImageNeighbors.length; i++) {
            BufferedImage thumb = thumbnails.get(queryImageNeighbors[i]);
            ImagePanelWithClass imgPan = new ImagePanelWithClass(classColors);
            imgPan.addMouseListener(new NeighborSelectionListener());
            imgPan.setImage(thumb, quantizedRepresentation.getLabelOf(
                    queryImageNeighbors[i]), queryImageNeighbors[i]);
            queryNNPanel.add(imgPan);
        }
        queryNNPanel.revalidate();
        queryNNPanel.repaint();
        // Update the predictions.
        if (trainedModels) {
            System.out.println("classifying");
            classifierPredictionsPanel.removeAll();
            classifierPredictionsPanel.revalidate();
            classifierPredictionsPanel.repaint();
            float[] trainingDists = new float[quantizedRepresentation.size()];
            for (int i = 0; i < queryImageNeighbors.length; i++) {
                trainingDists[queryImageNeighbors[i]] =
                        queryImageNeighborDists[i];
            }
            for (int i = 0; i < classifiers.length; i++) {
                System.out.println("classification by" + classifierNameList[i]);
                // Class affiliation prediction.
                float[] prediction;
                if (classifiers[i] instanceof
                        NeighborPointsQueryUserInterface) {
                    prediction = ((NeighborPointsQueryUserInterface) (
                            classifiers[i])).classifyProbabilistically(
                            queryImageRep, trainingDists, queryImageNeighbors);
                } else {
                    prediction = classifiers[i].classifyProbabilistically(
                            queryImageRep);
                }
                ClassifierResultPanel cResPanel = new ClassifierResultPanel();
                cResPanel.setResults(prediction, classifierNameList[i],
                        classColors, classNames);
                classifierPredictionsPanel.add(cResPanel);
            }
            classifierPredictionsPanel.revalidate();
            classifierPredictionsPanel.repaint();
        }
    } catch (Exception e) {
        System.err.println(e.getMessage());
    } finally {
        busyCalculating = false;
    }
}//GEN-LAST:event_reRankingButtonActionPerformed

    /**
     * Perform multi-dimensional scaling data visualization.
     *
     * @param evt ActionEvent object.
     */
private void mdsVisualizeItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_mdsVisualizeItemActionPerformed
    if (busyCalculating) {
        // If the system is busy calculating something, abort.
        return;
    }
    try {
        busyCalculating = true;
        // In the internal representation in this library, the distance matrix
        // is represented as upper triangular. However, the MDS component
        // from MDSJ requires a full square distance matrix, so we have to
        // generate it here from the more compact representation that is
        // normally used.
        float[][] halfDist = getDistances();
        if (halfDist == null || halfDist.length == 0) {
            busyCalculating = false;
            return;
        }
        double[][] fullDist = new double[halfDist.length][halfDist.length];
        for (int i = 0; i < halfDist.length; i++) {
            fullDist[i][i] = 0;
            for (int j = i + 1; j < halfDist.length; j++) {
                fullDist[i][j] = halfDist[i][j - i - 1];
                fullDist[j][i] = halfDist[i][j - i - 1];
            }
        }
        double[][] resultsReversed = MDSJ.classicalScaling(fullDist);
        System.out.println("MDS performed.");
        // Get the calculated image coordinates.
        imageCoordinatesXY = new float[halfDist.length][2];
        float maxX = -Float.MAX_VALUE;
        float maxY = -Float.MAX_VALUE;
        float minX = Float.MAX_VALUE;
        float minY = Float.MAX_VALUE;
        for (int i = 0; i < halfDist.length; i++) {
            imageCoordinatesXY[i][0] = (float) resultsReversed[0][i];
            imageCoordinatesXY[i][1] = (float) resultsReversed[1][i];
            if (imageCoordinatesXY[i][0] > maxX) {
                maxX = imageCoordinatesXY[i][0];
            }
            if (imageCoordinatesXY[i][0] < minX) {
                minX = imageCoordinatesXY[i][0];
            }
            if (imageCoordinatesXY[i][1] > maxY) {
                maxY = imageCoordinatesXY[i][1];
            }
            if (imageCoordinatesXY[i][1] < minY) {
                minY = imageCoordinatesXY[i][1];
            }
        }
        // Re-scale the result to fit the window.
        for (int i = 0; i < halfDist.length; i++) {
            imageCoordinatesXY[i][0] = ((imageCoordinatesXY[i][0] - minX)
                    / (maxX - minX)) * mdsCollectionPanel.getWidth();
            imageCoordinatesXY[i][1] = ((imageCoordinatesXY[i][1] - minY)
                    / (maxY - minY)) * mdsCollectionPanel.getHeight();
        }
        System.out.println("Coordinates calculated.");
        // Find the major hubs and calculate the size of their visual display.
        // Offsets in case some bounding rectangles fall out of the MDS panel.
        float offX, offY;
        mdsBackgrounds = new BufferedImage[50];
        if (highestHubnesses != null) {
            float maxFrequency = ArrayUtil.max(highestHubnesses[
                    neighborhoodSize - 1]);
            float[] thumbSizes =
                    new float[highestHubnesses[neighborhoodSize - 1].length];
            ArrayList<Rectangle2D> bounds = new ArrayList<>(thumbSizes.length);
            ArrayList<ImagePanelWithClass> imgsMDS =
                    new ArrayList<>(thumbSizes.length);
            for (int i = 0; i < thumbSizes.length; i++) {
                // Get the display thumbnail size.
                thumbSizes[i] =
                        pointScale(highestHubnesses[neighborhoodSize - 1][i],
                        maxFrequency, minImageScale, maxImageScale);
                // Calculate the offsets.
                if (imageCoordinatesXY[
                        highestHubIndexes[neighborhoodSize - 1][i]][0]
                        + thumbSizes[i] / 2 > mdsCollectionPanel.getWidth()) {
                    offX = (thumbSizes[i] / 2
                            - (mdsCollectionPanel.getWidth()
                            - imageCoordinatesXY[highestHubIndexes[
                            neighborhoodSize - 1][i]][0]));
                } else if (imageCoordinatesXY[highestHubIndexes[
                        neighborhoodSize - 1][i]][0] - thumbSizes[i] / 2 < 0) {
                    offX = imageCoordinatesXY[highestHubIndexes[
                            neighborhoodSize - 1][i]][0];
                } else {
                    offX = thumbSizes[i] / 2;
                }
                if (imageCoordinatesXY[highestHubIndexes[
                        neighborhoodSize - 1][i]][1] + thumbSizes[i] / 2
                        > mdsCollectionPanel.getHeight()) {
                    offY = (thumbSizes[i] / 2
                            - (mdsCollectionPanel.getHeight()
                            - imageCoordinatesXY[highestHubIndexes[
                            neighborhoodSize - 1][i]][1]));
                } else if (imageCoordinatesXY[highestHubIndexes[
                        neighborhoodSize - 1][i]][1] - thumbSizes[i] / 2 < 0) {
                    offY = imageCoordinatesXY[highestHubIndexes[
                            neighborhoodSize - 1][i]][1];
                } else {
                    offY = thumbSizes[i] / 2;
                }
                BufferedImage thumb = thumbnails.get(highestHubIndexes[
                        neighborhoodSize - 1][i]);
                ImagePanelWithClass imgPan = new ImagePanelWithClass(
                        classColors);
                imgPan.addMouseListener(new NeighborSelectionListener());
                imgPan.setImage(thumb, quantizedRepresentation.getLabelOf(
                        highestHubIndexes[neighborhoodSize - 1][i]),
                        highestHubIndexes[neighborhoodSize - 1][i]);
                imgsMDS.add(imgPan);
                bounds.add(new Rectangle2D.Float(imageCoordinatesXY[
                        highestHubIndexes[neighborhoodSize - 1][i]][0]
                        - offX, imageCoordinatesXY[highestHubIndexes[
                        neighborhoodSize - 1][i]][1] - offY,
                        thumbSizes[i], thumbSizes[i]));
            }
            mdsCollectionPanel.setImageSet(imgsMDS, bounds);
            // Set the background to the MDS screen.
            setMDSBackground();
            // Refresh.
            mdsCollectionPanel.revalidate();
            mdsCollectionPanel.repaint();
        }
    } catch (Exception e) {
        System.err.println(e.getMessage());
    } finally {
        busyCalculating = false;
    }
}//GEN-LAST:event_mdsVisualizeItemActionPerformed

    /**
     * Add the currently selected image to the kNN graph for visualization.
     *
     * @param evt ActionEvent object.
     */
    private void addSelectedButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addSelectedButtonActionPerformed
        addSelectedImageToGraph();
    }//GEN-LAST:event_addSelectedButtonActionPerformed

    /**
     * Add the nearest neighbors of the currently selected image to the kNN
     * graph for visualization.
     *
     * @param evt ActionEvent object.
     */
    private void addNNsButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addNNsButtonActionPerformed
        if (!this.neighborStatsCalculated || selectedImageHistory == null
                || selectedImageHistory.size() < 1) {
            return;
        }
        NeighborSetFinder nsf = getNSF();
        int[][] kneighbors = nsf.getKNeighbors();
        int index = selectedImageHistory.get(selectedImageIndexInHistory);
        int[] indexes = Arrays.copyOf(kneighbors[index], neighborhoodSize);
        addSelectedImagesToGraph(indexes);
    }//GEN-LAST:event_addNNsButtonActionPerformed

    /**
     * Add the reverse nearest neighbors of the currently selected image to the
     * kNN graph for visualization.
     *
     * @param evt ActionEvent object.
     */
    private void addRNNsButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addRNNsButtonActionPerformed
        if (!this.neighborStatsCalculated || selectedImageHistory == null
                || selectedImageHistory.size() < 1) {
            return;
        }
        int index = selectedImageHistory.get(selectedImageIndexInHistory);
        int[] indexes =
                new int[rnnSetsAllK[neighborhoodSize - 1][index].size()];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = rnnSetsAllK[neighborhoodSize - 1][index].get(i);
        }
        addSelectedImagesToGraph(indexes);
    }//GEN-LAST:event_addRNNsButtonActionPerformed

    /**
     * Remove the vertex from the kNN graph subset that is being visualized.
     *
     * @param evt ActionEvent object.
     */
    private void removeVertexButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_removeVertexButtonActionPerformed
        removeSelectedImageFromGraph();
    }//GEN-LAST:event_removeVertexButtonActionPerformed

    /**
     * This method loads the codebook profile from a file that contains the
     * visual word occurrence probabilities per class and is used for visual
     * word utility estimation.
     *
     * @param evt ActionEvent object.
     */
    private void loadCodebookProfileMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loadCodebookProfileMenuItemActionPerformed
        BufferedReader br;
        JFileChooser jfc = new JFileChooser(currentDirectory);
        if (codebook == null) {
            JOptionPane.showMessageDialog(this, "no codebook loaded", "Error",
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        jfc.setDialogTitle("Load Codebook Profiles: ");
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int rVal = jfc.showOpenDialog(ImageHubExplorer.this);
        if (rVal == JFileChooser.APPROVE_OPTION) {
            currentDirectory = jfc.getSelectedFile().getParentFile();
            codebookProfileFile = jfc.getSelectedFile();
            try {
                // Load the codebook profiles.
                br = new BufferedReader(new InputStreamReader(
                        new FileInputStream(codebookProfileFile)));
                int size = Integer.parseInt(br.readLine());
                if (size != codebook.getSize()) {
                    throw new Exception("codebook profile size not equal to "
                            + "codebook size");
                }
                codebookProfiles = new double[codebook.getSize()][numClasses];
                String line;
                String[] lineItems;
                float[] codeClassMax = new float[codebook.getSize()];
                float[] codeClassSums = new float[codebook.getSize()];
                for (int i = 0; i < size; i++) {
                    line = br.readLine();
                    lineItems = line.split(",");
                    if (lineItems.length != numClasses) {
                        throw new Exception("codebook profile class number "
                                + "inappropriate " + lineItems.length
                                + " instead of " + numClasses);
                    }
                    // Keep track of the stats.
                    codeClassMax[i] = 0;
                    codeClassSums[i] = 0;
                    for (int c = 0; c < numClasses; c++) {
                        codebookProfiles[i][c] = Double.parseDouble(
                                lineItems[c]);
                        codeClassMax[i] = (float) Math.max(codeClassMax[i],
                                codebookProfiles[i][c]);
                        codeClassSums[i] += codebookProfiles[i][c];
                    }
                }
                // Calculate the goodness of each visual word.
                codebookGoodness = new float[codebook.getSize()];
                for (int codeIndex = 0; codeIndex < codebookGoodness.length;
                        codeIndex++) {
                    if (codeClassSums[codeIndex] > 0) {
                        codebookGoodness[codeIndex] = codeClassMax[codeIndex]
                                / codeClassSums[codeIndex];
                    } else {
                        codebookGoodness[codeIndex] = 0;
                    }
                }
                float maxGoodness = ArrayUtil.max(codebookGoodness);
                float minGoodness = ArrayUtil.min(codebookGoodness);
                for (int codeIndex = 0; codeIndex < codebookGoodness.length;
                        codeIndex++) {
                    if ((maxGoodness - minGoodness) > 0) {
                        codebookGoodness[codeIndex] =
                                (codebookGoodness[codeIndex] - minGoodness)
                                / (maxGoodness - minGoodness);
                    }
                }
                codebookProfPanels =
                        new CodebookVectorProfilePanel[codebook.getSize()];
                for (int cInd = 0; cInd < codebook.getSize(); cInd++) {
                    CodebookVectorProfilePanel cProfPanel =
                            new CodebookVectorProfilePanel();
                    cProfPanel.setResults(codebookProfiles[cInd], cInd,
                            classColors, classNames);
                    cProfPanel.setPreferredSize(new Dimension(120, 120));
                    cProfPanel.setMinimumSize(new Dimension(120, 120));
                    cProfPanel.setMaximumSize(new Dimension(120, 120));
                    codebookProfPanels[cInd] = cProfPanel;
                }
                JOptionPane.showMessageDialog(frameReference, "Load completed");
            } catch (Exception e) {
                System.err.println(e.getMessage());
                codebookProfiles = null;
                codebookGoodness = null;
            }
        }
    }//GEN-LAST:event_loadCodebookProfileMenuItemActionPerformed

    /**
     * This method calculates the primary Tanimoto distances.
     *
     * @param evt ActionEvent object.
     */
    private void tanimotoMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_tanimotoMenuItemActionPerformed
        if (!busyCalculating) {
            // If the system is currently performing calculations, abort the
            // call.
            busyCalculating = true;
            try {
                primaryCMet = CombinedMetric.FLOAT_TANIMOTO;
                primaryDMatFile = new File(workspace, "distancesNNSets"
                        + File.separator
                        + primaryCMet.getFloatMetric().getClass().getName()
                        + File.separator + "dMat.txt");
                boolean calcNSets = !(new File(primaryDMatFile.getParentFile(),
                        "knnSets.txt")).exists();
                if (quantizedRepresentation != null
                        && !quantizedRepresentation.isEmpty()
                        && !primaryDMatFile.exists()) {
                    // If they haven't already been calculated, calculate them
                    // now.
                    Thread t = new Thread(
                            new DistanceCalcHelperPrimary(calcNSets,
                            primaryDMatFile.getParentFile()));
                    t.start();
                } else if (primaryDMatFile.exists()) {
                    // If they have already been calculated, just load them.
                    loadDistancesAndNeighbors(primaryDMatFile, PRIMARY_METRIC);
                }
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }//GEN-LAST:event_tanimotoMenuItemActionPerformed

    /**
     * This method calculates the symmetrized Kullback-Leibler divergence as a
     * distance measure.
     *
     * @param evt ActionEvent object.
     */
    private void klMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_klMenuItemActionPerformed
        if (!busyCalculating) {
            // If the system is currently performing calculations, abort the
            // call.
            busyCalculating = true;
            try {
                primaryCMet = CombinedMetric.FLOAT_KL;
                primaryDMatFile = new File(workspace, "distancesNNSets"
                        + File.separator
                        + primaryCMet.getFloatMetric().getClass().getName()
                        + File.separator + "dMat.txt");
                boolean calcNSets = !(new File(primaryDMatFile.getParentFile(),
                        "knnSets.txt")).exists();
                if (quantizedRepresentation != null
                        && !quantizedRepresentation.isEmpty()
                        && !primaryDMatFile.exists()) {
                    // If they haven't already been calculated, calculate them
                    // now.
                    Thread t = new Thread(new DistanceCalcHelperPrimary(
                            calcNSets, primaryDMatFile.getParentFile()));
                    t.start();
                } else if (primaryDMatFile.exists()) {
                    // If they have already been calculated, just load them.
                    loadDistancesAndNeighbors(primaryDMatFile, PRIMARY_METRIC);
                }
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }//GEN-LAST:event_klMenuItemActionPerformed

    /**
     * Calculate the primary Bray-Curtis distances.
     *
     * @param evt ActionEvent object.
     */
    private void bcMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_bcMenuItemActionPerformed
        if (!busyCalculating) {
            // If the system is currently performing calculations, abort the
            // call.
            busyCalculating = true;
            try {
                primaryCMet = CombinedMetric.FLOAT_BRAY_CURTIS;
                primaryDMatFile = new File(workspace, "distancesNNSets"
                        + File.separator
                        + primaryCMet.getFloatMetric().getClass().getName()
                        + File.separator + "dMat.txt");
                boolean calcNSets = !(new File(primaryDMatFile.getParentFile(),
                        "knnSets.txt")).exists();
                if (quantizedRepresentation != null
                        && !quantizedRepresentation.isEmpty()
                        && !primaryDMatFile.exists()) {
                    // If they haven't already been calculated, calculate them
                    // now.
                    Thread t = new Thread(new DistanceCalcHelperPrimary(
                            calcNSets, primaryDMatFile.getParentFile()));
                    t.start();
                } else if (primaryDMatFile.exists()) {
                    // If they have already been calculated, just load them.
                    loadDistancesAndNeighbors(primaryDMatFile, PRIMARY_METRIC);
                }
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }//GEN-LAST:event_bcMenuItemActionPerformed

    /**
     * Calculate the primary Canberra distances.
     *
     * @param evt ActionEvent object.
     */
    private void canMenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_canMenuItemActionPerformed
        if (!busyCalculating) {
            // If the system is currently performing calculations, abort the
            // call.
            busyCalculating = true;
            try {
                primaryCMet = CombinedMetric.FLOAT_CANBERRA;
                primaryDMatFile = new File(workspace, "distancesNNSets"
                        + File.separator
                        + primaryCMet.getFloatMetric().getClass().getName()
                        + File.separator + "dMat.txt");
                boolean calcNSets = !(new File(primaryDMatFile.getParentFile(),
                        "knnSets.txt")).exists();
                if (quantizedRepresentation != null
                        && !quantizedRepresentation.isEmpty()
                        && !primaryDMatFile.exists()) {
                    // If they haven't already been calculated, calculate them
                    // now.
                    Thread t = new Thread(new DistanceCalcHelperPrimary(
                            calcNSets, primaryDMatFile.getParentFile()));
                    t.start();
                } else if (primaryDMatFile.exists()) {
                    // If they have already been calculated, just load them.
                    loadDistancesAndNeighbors(primaryDMatFile, PRIMARY_METRIC);
                }
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }//GEN-LAST:event_canMenuItemActionPerformed

    /**
     * Get a screen capture of the kNN graph visualization and save the image to
     * a file.
     *
     * @param evt ActionEvent object.
     */
    private void graphScreenCaptureItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_graphScreenCaptureItemActionPerformed
        if (neighborGraphScrollPane != null) {
            BufferedImage bufImage = ScreenImage.createImage(
                    (JComponent) neighborGraphScrollPane.getViewport().
                    getComponent(0));
            try {
                File outFile;
                JFileChooser jfc = new JFileChooser(currentDirectory);
                jfc.setDialogTitle("Select file to save the component image: ");
                jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
                int rVal = jfc.showOpenDialog(ImageHubExplorer.this);
                if (rVal == JFileChooser.APPROVE_OPTION) {
                    currentDirectory = jfc.getSelectedFile().getParentFile();
                    outFile = jfc.getSelectedFile();
                    ImageIO.write(bufImage, "jpg", outFile);
                }
            } catch (Exception e) {
                System.err.println("problem writing file: " + e.getMessage());
            }
        }
    }//GEN-LAST:event_graphScreenCaptureItemActionPerformed

    /**
     * Get a screen capture of the MDS data visualization and save the image to
     * a file.
     *
     * @param evt ActionEvent object.
     */
    private void mdsScreenCaptureItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_mdsScreenCaptureItemActionPerformed
        if (mdsCollectionPanel != null) {
            BufferedImage bufImage =
                    ScreenImage.createImage((JComponent) mdsCollectionPanel);
            try {
                File outFile;
                JFileChooser jfc = new JFileChooser(currentDirectory);
                jfc.setDialogTitle("Select file to save the component image: ");
                jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
                int rVal = jfc.showOpenDialog(ImageHubExplorer.this);
                if (rVal == JFileChooser.APPROVE_OPTION) {
                    currentDirectory = jfc.getSelectedFile().getParentFile();
                    outFile = jfc.getSelectedFile();
                    ImageIO.write(bufImage, "jpg", outFile);
                }
            } catch (Exception e) {
                System.err.println("problem writing file: " + e.getMessage());
            }
        }
    }//GEN-LAST:event_mdsScreenCaptureItemActionPerformed

    /**
     * Remove all nodes from the kNN subgraph visualizations.
     *
     * @param evt ActionEvent object.
     */
    private void removeAllButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_removeAllButtonActionPerformed
        if (neighborStatsCalculated && neighborGraphs != null
                && selectedImageHistory != null
                && selectedImageHistory.size() > 0) {
            neighborGraphs = new DirectedGraph[50];
            graphsInit();
            System.gc();
            neighborGraphScrollPane.revalidate();
            neighborGraphScrollPane.repaint();
        }
    }//GEN-LAST:event_removeAllButtonActionPerformed

    /**
     * This method initiates the visual word utility assessment taking the
     * currently selected image as an example for examining the visual word
     * distribution and the utility of different image regions. It loads the
     * features from the disk if available and then starts the
     * QuantizedImageViewer component.
     *
     * @param evt ActionEvent object.
     */
    private void selSIFTmenuItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_selSIFTmenuItemActionPerformed
        if (selectedImageHistory == null
                || selectedImageIndexInHistory >= selectedImageHistory.size()) {
            return;
        }
        // Ensure the codebook is properly loaded and the profiles are
        // available.
        if (codebook == null || codebookProfiles == null) {
            System.out.println("Both codebook and codebook profiles need to be"
                    + " loaded");
            return;
        }
        // Get the image index within the image data.
        int index = selectedImageHistory.get(selectedImageIndexInHistory);
        String shortPath = imgPaths.get(index).substring(
                (new File(workspace, "photos")).getPath().length(),
                imgPaths.get(index).length());
        File repDir = new File(workspace, "representation");
        File lFeatDir = new File(repDir, "raw_representation");
        int dotIndex = shortPath.lastIndexOf(".");
        String shortPathCutOff = shortPath.substring(0, dotIndex);
        // For files in SiftWin keypoint format.
        String shortPathKey = shortPathCutOff + ".key";
        // For files in the OpenCV format.
        String shortPathKeypoint = shortPathCutOff + ".kp";
        String shortPathDescriptor = shortPathCutOff + ".desc";
        LFeatRepresentation lFeatRep = null;
        File imgSIFTFile = new File(lFeatDir, shortPathKey);
        if (imgSIFTFile.exists()) {
            try {
                // Load the features from the disk
                lFeatRep = SiftUtil.importFeaturesFromSift(imgSIFTFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        } else {
            File imgKPFile = new File(lFeatDir, shortPathKeypoint);
            File imgDescFile = new File(lFeatDir, shortPathDescriptor);
            if (!imgKPFile.exists() || !imgDescFile.exists()) {
                System.err.println("No supported local feature format detected"
                        + "for the image.");
            } else {
                try {
                    // Load the features from the disk
                    lFeatRep = OpenCVFeatureIO.loadImageRepresentation(
                            imgKPFile, imgDescFile);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        }
        if (lFeatRep == null || lFeatRep.isEmpty()) {
            return;
        }
        // Start examining the utility of different visual words on the image.
        BufferedImage originalImage = getPhoto(index);
        QuantizedImageViewer qiv = new QuantizedImageViewer(
                originalImage, lFeatRep, codebookGoodness, codebook,
                codebookProfiles, codebookProfPanels, classColors, classNames);
        qiv.setVisible(true);
    }//GEN-LAST:event_selSIFTmenuItemActionPerformed

    /**
     * This method determines the image thumbnail scale based on its occurrence
     * frequency.
     *
     * @param numOcc Float that is the image neighbor occurrence frequency.
     * @param MaxOcc Float that is the maximum occurrence frequency.
     * @param minScale Integer that is the minimal scale.
     * @param maxScale Integer that is the maximal scale.
     * @return Float that is the appropriate scale for the image thumbnail.
     * @throws Exception
     */
    private static float pointScale(float numOcc, float MaxOcc, int minScale,
            int maxScale) throws Exception {
        return minScale + (maxScale - minScale)
                * (float) Math.pow(((numOcc) / (MaxOcc)), 0.5);
    }

    /**
     * This method calculates and sets the background landscape for the MDS data
     * visualization. The landscape is set according the the good/bad hubness
     * densities of different regions in the panel, based on the MDS projection
     * of the data. The landscape is then softened by applying some blurring.
     */
    public void setMDSBackground() {
        if (mdsBackgrounds[neighborhoodSize - 1] == null) {
            // Get the width and height of the panel.
            int width = mdsCollectionPanel.getWidth();
            int height = mdsCollectionPanel.getHeight();
            // Cell size.
            int step = 100;
            int steppedWidth = width / step;
            if (width % step != 0) {
                steppedWidth++;
            }
            int steppedHeight = height / step;
            if (height % step != 0) {
                steppedHeight++;
            }
            // Get the kNN sets for the appropriate neighborhood size.
            NeighborSetFinder nsf = getNSF().copy();
            nsf.recalculateStatsForSmallerK(neighborhoodSize);
            int[] badHubness = nsf.getBadFrequencies();
            int[] goodHubness = nsf.getGoodFrequencies();
            int bucketX, bucketY;
            // Place all the projected data into the appropriate buckets in the
            // cell grid.
            // First initialize the buckets.
            ArrayList<Integer>[][] bucketedData =
                    new ArrayList[steppedWidth][steppedHeight];
            for (int i = 0; i < steppedWidth; i++) {
                for (int j = 0; j < steppedHeight; j++) {
                    bucketedData[i][j] = new ArrayList<>(5);
                }
            }
            // Insert the data into the buckets.
            for (int i = 0; i < imageCoordinatesXY.length; i++) {
                bucketX = (int) (imageCoordinatesXY[i][0] / step);
                bucketY = (int) (imageCoordinatesXY[i][1] / step);
                if (bucketX >= bucketedData.length) {
                    continue;
                }
                if (bucketY >= bucketedData[bucketX].length) {
                    continue;
                }
                bucketedData[bucketX][bucketY].add(i);
            }
            int[] landscapeRaster = new int[width * height];
            // The goodness of a pixel.
            int greennessValue;
            // Good and bad hubness contributions.
            double ghFactor;
            double bhFactor;
            // Weight of a contribution.
            double weight;
            // Kernel width.
            double sigma = 0.05;
            int pX;
            int pY;
            int pIndex;
            for (int i = 0; i < landscapeRaster.length; i++) {
                pX = i % width;
                pY = i / width;
                bucketX = (pX) / step;
                bucketY = (pY) / step;
                ghFactor = 0;
                bhFactor = 0;
                for (int j = 0; j < bucketedData[bucketX][bucketY].size();
                        j++) {
                    pIndex = bucketedData[bucketX][bucketY].get(j);
                    weight = Math.min(Math.exp(-sigma
                            * ((imageCoordinatesXY[pIndex][0] - pX)
                            * (imageCoordinatesXY[pIndex][0] - pX)
                            + (imageCoordinatesXY[pIndex][1] - pY)
                            * (imageCoordinatesXY[pIndex][1] - pY))), 1);
                    ghFactor += weight * goodHubness[pIndex];
                    bhFactor += weight * badHubness[pIndex];
                }
                if (ghFactor > 0 || bhFactor > 0) {
                    greennessValue = (int) (255
                            * ((ghFactor) / (ghFactor + bhFactor)));
                    landscapeRaster[i] = greennessValue << 8
                            | (255 - greennessValue) << 16;
                } else {
                    landscapeRaster[i] = 0x809080;
                }
            }
            // Now perform several passes of box blur to soften the landscape.
            BoxBlur bb = new BoxBlur(18);
            bb.blurPixels(landscapeRaster, new int[landscapeRaster.length],
                    new Dimension(width, height));
            bb = new BoxBlur(9);
            bb.blurPixels(landscapeRaster, new int[landscapeRaster.length],
                    new Dimension(width, height));
            bb = new BoxBlur(17);
            bb.blurPixels(landscapeRaster, new int[landscapeRaster.length],
                    new Dimension(width, height));
            BufferedImage bckg = new BufferedImage(width, height,
                    BufferedImage.TYPE_INT_RGB);
            bckg.setRGB(0, 0, width, height, landscapeRaster, 0, width);
            mdsBackgrounds[neighborhoodSize - 1] = bckg;
            mdsCollectionPanel.setBackgroundImage(bckg);
            // Refresh the display.
            mdsCollectionPanel.revalidate();
            mdsCollectionPanel.repaint();
        } else {
            // Load a landscape that was already calculated.
            mdsCollectionPanel.setBackgroundImage(
                    mdsBackgrounds[neighborhoodSize - 1]);
            mdsCollectionPanel.revalidate();
            mdsCollectionPanel.repaint();
        }

    }

    /**
     * This class calculates the kNN and hubness-related stats for a general
     * data overview that are shown in the default visualization screen.
     */
    private class NSFStatsHelper implements Runnable {

        /**
         * The default constructor.
         */
        public NSFStatsHelper() {
        }

        @Override
        public void run() {
            if (busyCalculating) {
                // If the system is currently performing calculations, abort the
                // call.
                return;
            }
            busyCalculating = true;
            try {
                // Get the kNN sets.
                NeighborSetFinder nsf = getNSF();
                int k_max = 50;
                // Individual stats are calculates by the specialized objects.
                // Here is their initialization.
                HubnessAboveThresholdExplorer hte =
                        new HubnessAboveThresholdExplorer(1, true, nsf);
                HubnessSkewAndKurtosisExplorer hske =
                        new HubnessSkewAndKurtosisExplorer(nsf);
                HubnessExtremesGrabber heg =
                        new HubnessExtremesGrabber(true, nsf);
                KNeighborEntropyExplorer knee =
                        new KNeighborEntropyExplorer(nsf, numClasses);
                HubOrphanRegularPercentagesCalculator hubOrph =
                        new HubOrphanRegularPercentagesCalculator(
                        nsf, k_max);
                BucketedOccDistributionGetter bucketGetter =
                        new BucketedOccDistributionGetter(nsf, 50, 1);
                bucketedOccurrenceDistributions =
                        bucketGetter.getBucketedDistributions();
                // Get the percentage of elements that occurs at least once, for
                // each neighborhood size.
                aboveZeroArray = hte.getThresholdPercentageArray();
                // Get the neighbor occurrence distribution properties.
                hske.calcSkewAndKurtosisArrays();
                skewArray = hske.getOccFreqsSkewnessArray();
                kurtosisArray = hske.getOccFreqsKurtosisArray();
                highestHubnesses = heg.getHubnessExtremesForKValues(
                        numImagesDrawn);
                highestHubIndexes = heg.getExtremeIndexes();
                // Get the entropies of direct and reverse neighbor sets.
                knee.calculateAllKNNEntropyStats();
                kEntropies = knee.getDirectEntropyMeans();
                reverseKNNEntropies = knee.getReverseEntropyMeans();
                kEntropySkews = knee.getDirectEntropySkews();
                reverseKNNEntropySkews = knee.getReverseEntropySkews();
                badHubnessArray = nsf.getLabelMismatchPercsAllK();
                // Get the percentage of hubs, regulars and orphans in the data.
                hubOrph.calculatePTypePercs();
                hubPercs = hubOrph.getHubPercs();
                orphanPercs = hubOrph.getOrphanPercs();
                regularPercs = hubOrph.getRegularPercs();
                // Get the global class-to-class hubness distribution.
                globalClassToClasshubness = new float[k_max][][];
                for (int kTmp = 1; kTmp <= k_max; kTmp++) {
                    globalClassToClasshubness[kTmp - 1] =
                            nsf.getGlobalClassToClassForKforFuzzy(kTmp,
                            numClasses, 0.01f, true);
                }
                // Prepare the bucketed distribution data for display in charts.
                DefaultCategoryDataset hDistDataset =
                        new DefaultCategoryDataset();
                for (int i = 0; i
                        < bucketedOccurrenceDistributions[
                        neighborhoodSize - 1].length; i++) {
                    hDistDataset.addValue(bucketedOccurrenceDistributions[
                            neighborhoodSize - 1][i],
                            "Number of Examples", i + "");
                }
                JFreeChart chart = ChartFactory.createBarChart(
                        "Occurrence Frequency Distribution", "", "",
                        hDistDataset, PlotOrientation.VERTICAL, false, true,
                        false);
                ChartPanel chartPanel = new ChartPanel(chart);
                chartPanel.setPreferredSize(new Dimension(440, 180));
                // Refresh the display.
                chartHoldingPanelOccDistribution.removeAll();
                chartHoldingPanelOccDistribution.add(chartPanel);
                chartHoldingPanelOccDistribution.revalidate();
                chartHoldingPanelOccDistribution.repaint();
                classesScrollPanel.setPreferredSize(new Dimension(760,
                        390 * numClasses));
                classesScrollPanel.setMaximumSize(new Dimension(760,
                        390 * numClasses));
                classesScrollPanel.setMinimumSize(new Dimension(760,
                        390 * numClasses));
                classesScrollPanel.removeAll();
                // Per-class top hubs, good hubs and bad hubs.
                classTopHubLists = new ArrayList[50][numClasses];
                classTopGoodHubsList = new ArrayList[50][numClasses];
                classTopBadHubsList = new ArrayList[50][numClasses];
                classHubnessArrValues = new ArrayList[50][numClasses];
                classHubnessArrGoodValues = new ArrayList[50][numClasses];
                classHubnessArrBadValues = new ArrayList[50][numClasses];
                // Make lists of instances belonging to each class.
                classImageIndexes = new ArrayList[numClasses];
                for (int c = 0; c < numClasses; c++) {
                    classImageIndexes[c] = new ArrayList<>(
                            quantizedRepresentation.size());
                }
                int[][] permutationTotalHubness = new int[numClasses][];
                int[][] permutationGoodHubness = new int[numClasses][];
                int[][] permutationBadHubness = new int[numClasses][];
                int label;
                // Populate the class lists.
                for (int i = 0; i < quantizedRepresentation.size(); i++) {
                    label = quantizedRepresentation.getLabelOf(i);
                    classImageIndexes[label].add(i);
                }
                NeighborSetFinder nsfSmall = nsf.copy();
                rnnSetsAllK = new ArrayList[50][quantizedRepresentation.size()];
                occurrenceProfilesAllK = new float[50][
                        quantizedRepresentation.size()][numClasses];
                for (int kTmp = 50; kTmp > 0; kTmp--) {
                    // For each neighborhood size.
                    nsfSmall.recalculateStatsForSmallerK(kTmp);
                    occurrenceProfilesAllK[kTmp - 1] =
                            nsfSmall.getDataClassNeighborRelationNonNormalized(
                            neighborhoodSize, numClasses, false);
                    // Populate the reverse neighbor sets.
                    for (int i = 0; i < quantizedRepresentation.size(); i++) {
                        rnnSetsAllK[kTmp - 1][i] = new ArrayList<>(
                                nsfSmall.getReverseNeighbors()[i].size());
                        for (int j = 0; j < nsfSmall.
                                getReverseNeighbors()[i].size(); j++) {
                            rnnSetsAllK[kTmp - 1][i].add(
                                    nsfSmall.getReverseNeighbors()[i].get(j));
                        }
                    }
                    // Initialize the lists of top hubs, good hubs and bad hubs
                    // for each class.
                    for (int c = 0; c < numClasses; c++) {
                        classTopHubLists[kTmp - 1][c] =
                                new ArrayList<>(quantizedRepresentation.size());
                        classTopGoodHubsList[kTmp - 1][c] =
                                new ArrayList<>(quantizedRepresentation.size());
                        classTopBadHubsList[kTmp - 1][c] =
                                new ArrayList<>(quantizedRepresentation.size());
                        classHubnessArrValues[kTmp - 1][c] = new ArrayList<>(
                                quantizedRepresentation.size());
                        classHubnessArrGoodValues[kTmp - 1][c] =
                                new ArrayList<>(quantizedRepresentation.size());
                        classHubnessArrBadValues[kTmp - 1][c] =
                                new ArrayList<>(quantizedRepresentation.size());
                    }
                    // Get the good, bad and total hubness for each data
                    // instance.
                    int[] goodHubness = nsfSmall.getGoodFrequencies();
                    int[] badHubness = nsfSmall.getBadFrequencies();
                    int[] totalHubness = nsfSmall.getNeighborFrequencies();
                    // Insert the occurrence values into the class lists.
                    for (int i = 0; i < quantizedRepresentation.size(); i++) {
                        label = quantizedRepresentation.getLabelOf(i);
                        classHubnessArrValues[kTmp - 1][label].add(
                                totalHubness[i]);
                        classHubnessArrGoodValues[kTmp - 1][label].add(
                                goodHubness[i]);
                        classHubnessArrBadValues[kTmp - 1][label].add(
                                badHubness[i]);
                    }
                    // Sort the class lists of good, bad and total hubness.
                    for (int c = 0; c < numClasses; c++) {
                        permutationTotalHubness[c] = AuxSort.sortIIndexedValue(
                                classHubnessArrValues[kTmp - 1][c], true);
                        permutationGoodHubness[c] = AuxSort.sortIIndexedValue(
                                classHubnessArrGoodValues[kTmp - 1][c], true);
                        permutationBadHubness[c] = AuxSort.sortIIndexedValue(
                                classHubnessArrBadValues[kTmp - 1][c], true);
                    }
                    // Populate the class lists of top total, good and bad hubs.
                    for (int c = 0; c < numClasses; c++) {
                        for (int i = 0; i < classImageIndexes[c].size(); i++) {
                            classTopHubLists[kTmp - 1][c].add(
                                    classImageIndexes[c].get(
                                    permutationTotalHubness[c][i]));
                            classTopGoodHubsList[kTmp - 1][c].add(
                                    classImageIndexes[c].get(
                                    permutationGoodHubness[c][i]));
                            classTopBadHubsList[kTmp - 1][c].add(
                                    classImageIndexes[c].get(
                                    permutationBadHubness[c][i]));
                        }
                    }
                }
                // Calculate the distribution of point types in each class.
                int[][] kneighbors = nsf.getKNeighbors();
                classPTypes = new float[numClasses][4];
                int labelMatchCount;
                for (int i = 0; i < quantizedRepresentation.size(); i++) {
                    label = quantizedRepresentation.getLabelOf(i);
                    labelMatchCount = 0;
                    for (int j = 0; j < 5; j++) {
                        if (quantizedRepresentation.
                                getLabelOf(kneighbors[i][j]) == label) {
                            labelMatchCount++;
                        }
                    }
                    switch (labelMatchCount) {
                        case 0:
                            // Outlier.
                            classPTypes[label][3]++;
                            break;
                        case 1:
                            // Rare point.
                            classPTypes[label][2]++;
                            break;
                        case 2:
                            // Borderline.
                            classPTypes[label][1]++;
                            break;
                        case 3:
                            // Borderline.
                            classPTypes[label][1]++;
                            break;
                        case 4:
                            // Safe.
                            classPTypes[label][0]++;
                            break;
                        case 5:
                            // Safe.
                            classPTypes[label][0]++;
                            break;
                        default:
                            classPTypes[label][0]++;
                    }

                }
                // Update the graphical elements.
                classStatsOverviews = new ClassHubsPanel[numClasses];
                for (int c = 0; c < numClasses; c++) {
                    ClassHubsPanel chp =
                            new ClassHubsPanel(classColors[c], classNames[c]);
                    chp.setPointTypeDistribution(classPTypes[c]);
                    chp.revalidate();
                    chp.repaint();
                    classStatsOverviews[c] = chp;
                    JPanel hubsPanel = chp.getHubsPanel();
                    JPanel hubsPanelGood = chp.getGoodHubsPanel();
                    JPanel hubsPanelBad = chp.getBadHubsPanel();
                    // Handle the major hubs in the class.
                    for (int i = 0; i < Math.min(50,
                            classImageIndexes[c].size()); i++) {
                        BufferedImage thumb = thumbnails.get(
                                classTopHubLists[
                                neighborhoodSize - 1][c].get(i));
                        ImagePanelWithClass imgPan =
                                new ImagePanelWithClass(classColors);
                        imgPan.addMouseListener(
                                new NeighborSelectionListener());
                        imgPan.setImage(thumb,
                                quantizedRepresentation.getLabelOf(
                                classTopHubLists[
                                neighborhoodSize - 1][c].get(i)),
                                classTopHubLists[
                                neighborhoodSize - 1][c].get(i));
                        hubsPanel.add(imgPan);
                    }
                    hubsPanel.revalidate();
                    hubsPanel.repaint();
                    // Handle the major good hubs in the class.
                    for (int i = 0; i < Math.min(50,
                            classImageIndexes[c].size()); i++) {
                        BufferedImage thumb = thumbnails.get(
                                classTopGoodHubsList[
                                neighborhoodSize - 1][c].get(i));
                        ImagePanelWithClass imgPan =
                                new ImagePanelWithClass(classColors);
                        imgPan.addMouseListener(
                                new NeighborSelectionListener());
                        imgPan.setImage(thumb,
                                quantizedRepresentation.getLabelOf(
                                classTopGoodHubsList[
                                neighborhoodSize - 1][c].get(i)),
                                classTopGoodHubsList[
                                neighborhoodSize - 1][c].get(i));
                        hubsPanelGood.add(imgPan);
                    }
                    hubsPanelGood.revalidate();
                    hubsPanelGood.repaint();
                    // Handle the major bad hubs in the class.
                    for (int i = 0; i < Math.min(50,
                            classImageIndexes[c].size()); i++) {
                        BufferedImage thumb = thumbnails.get(
                                classTopBadHubsList[
                                neighborhoodSize - 1][c].get(i));
                        ImagePanelWithClass imgPan =
                                new ImagePanelWithClass(classColors);
                        imgPan.addMouseListener(
                                new NeighborSelectionListener());
                        imgPan.setImage(thumb,
                                quantizedRepresentation.getLabelOf(
                                classTopBadHubsList[
                                neighborhoodSize - 1][c].get(i)),
                                classTopBadHubsList[
                                neighborhoodSize - 1][c].get(i));
                        hubsPanelBad.add(imgPan);
                    }
                    hubsPanelBad.revalidate();
                    hubsPanelBad.repaint();
                    chp.revalidate();
                    chp.repaint();
                    classesScrollPanel.add(chp);
                }
                // Refresh the display.
                classesScrollPanel.revalidate();
                classesScrollPanel.repaint();
                classesScrollPane.revalidate();
                classesScrollPane.repaint();
                neighborStatsCalculated = true;
                graphsInit();
                JOptionPane.showMessageDialog(frameReference,
                        "Neighbor Stats calculated.");
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * This class calculates the secondary NICDM distances.
     */
    private class NICDMHelper implements Runnable {

        /**
         * The default constructor.
         */
        public NICDMHelper() {
        }

        @Override
        public void run() {
            if (busyCalculating) {
                // If the system is currently performing calculations, abort the
                // call.
                return;
            }
            busyCalculating = true;
            try {
                File pDir = primaryDMatFile.getParentFile();
                File sDir = new File(pDir, "secondary" + File.separator
                        + "nicdm");
                File matFile = new File(sDir, "dMat.txt");
                if (matFile.exists() && secondaryLoadFlag) {
                    // If it already exists and the flag allows the load, load
                    // from the disk.
                    distMatrixSecondary = loadDMatFromFile(matFile);
                } else {
                    // Calculate the secondary distance matrix.
                    secondaryCMet = new NICDMCalculator(nsfPrimary);
                    distMatrixSecondary = ((NICDMCalculator) secondaryCMet).
                            getTransformedDMatFromNSFPrimaryDMat();
                }
                // Load or calculate the secondary kNN sets.
                File neighborsFile = new File(sDir, "knnSets.txt");
                if (neighborsFile.exists() && secondaryLoadFlag) {
                    nsfSecondary = NeighborSetFinder.loadNSF(neighborsFile,
                            quantizedRepresentation);
                    nsfSecondary.setDistances(distMatrixSecondary);
                } else {
                    nsfSecondary = new NeighborSetFinder(
                            quantizedRepresentation, distMatrixSecondary,
                            secondaryCMet);
                    nsfSecondary.calculateNeighborSetsMultiThr(50, 8);
                }
                printDMatToFile(distMatrixSecondary, matFile);
                nsfSecondary.saveNeighborSets(neighborsFile);
                neighborStatsCalculated = false;
                imageCoordinatesXY = null;
                JOptionPane.showMessageDialog(frameReference,
                        "NICDM calculated");
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * This class calculates the secondary local scaling distances.
     */
    private class LocalScalingHelper implements Runnable {

        /**
         * The defautl constructor.
         */
        public LocalScalingHelper() {
        }

        @Override
        public void run() {
            if (busyCalculating) {
                // If the system is currently performing calculations, abort the
                // call.
                return;
            }
            busyCalculating = true;
            try {
                File pDir = primaryDMatFile.getParentFile();
                File sDir = new File(pDir, "secondary" + File.separator
                        + "localscaling");
                File matFile = new File(sDir, "dMat.txt");
                if (matFile.exists() && secondaryLoadFlag) {
                    // If it already exists and the flag allows the load, load
                    // from the disk.
                    distMatrixSecondary = loadDMatFromFile(matFile);
                } else {
                    // Calculate the secondary distance matrix.
                    secondaryCMet = new LocalScalingCalculator(nsfPrimary);
                    distMatrixSecondary = (
                            (LocalScalingCalculator) secondaryCMet).
                            getTransformedDMatFromNSFPrimaryDMat();
                }
                // Load or calculate the secondary kNN sets.
                File neighborsFile = new File(sDir, "knnSets.txt");
                if (neighborsFile.exists() && secondaryLoadFlag) {
                    nsfSecondary = NeighborSetFinder.loadNSF(neighborsFile,
                            quantizedRepresentation);
                    nsfSecondary.setDistances(distMatrixSecondary);
                } else {
                    nsfSecondary = new NeighborSetFinder(
                            quantizedRepresentation, distMatrixSecondary,
                            secondaryCMet);
                    nsfSecondary.calculateNeighborSetsMultiThr(50, 8);
                }
                printDMatToFile(distMatrixSecondary, matFile);
                nsfSecondary.saveNeighborSets(neighborsFile);
                neighborStatsCalculated = false;
                imageCoordinatesXY = null;
                JOptionPane.showMessageDialog(frameReference,
                        "Local scaling calculated");
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * This class calculates the secondary mutual proximity distances.
     */
    private class MutualProximityHelper implements Runnable {

        /**
         * The default constructor.
         */
        public MutualProximityHelper() {
        }

        @Override
        public void run() {
            if (busyCalculating) {
                // If the system is currently performing calculations, abort the
                // call.
                return;
            }
            busyCalculating = true;
            try {
                File pDir = primaryDMatFile.getParentFile();
                File sDir = new File(pDir, "secondary" + File.separator
                        + "mutualprox");
                File matFile = new File(sDir, "dMat.txt");
                if (matFile.exists() && secondaryLoadFlag) {
                    // If it already exists and the flag allows the load, load
                    // from the disk.
                    distMatrixSecondary = loadDMatFromFile(matFile);
                } else {
                    // Calculate the secondary distance matrix.
                    secondaryCMet = new MutualProximityCalculator(
                            distMatrixPrimary, quantizedRepresentation,
                            primaryCMet);
                    distMatrixSecondary = (
                            (MutualProximityCalculator) secondaryCMet).
                            getTransformedDMat();
                }
                // Load or calculate the secondary kNN sets.
                File neighborsFile = new File(sDir, "knnSets.txt");
                if (neighborsFile.exists() && secondaryLoadFlag) {
                    nsfSecondary = NeighborSetFinder.loadNSF(neighborsFile,
                            quantizedRepresentation);
                    nsfSecondary.setDistances(distMatrixSecondary);
                } else {
                    nsfSecondary = new NeighborSetFinder(
                            quantizedRepresentation, distMatrixSecondary,
                            secondaryCMet);
                    nsfSecondary.calculateNeighborSetsMultiThr(50, 8);
                }
                printDMatToFile(distMatrixSecondary, matFile);
                nsfSecondary.saveNeighborSets(neighborsFile);
                neighborStatsCalculated = false;
                imageCoordinatesXY = null;
                JOptionPane.showMessageDialog(frameReference,
                        "Mutual proximity calculated");
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * This class calculates the secondary simhub distances, that are the
     * hubness-aware version of simcos distances.
     */
    private class SimHubHelper implements Runnable {

        /**
         * The default constructor.
         */
        public SimHubHelper() {
        }

        @Override
        public void run() {
            if (busyCalculating) {
                // If the system is currently performing calculations, abort the
                // call.
                return;
            }
            busyCalculating = true;
            try {
                File pDir = primaryDMatFile.getParentFile();
                File sDir = new File(pDir, "secondary" + File.separator
                        + "simhub");
                File matFile = new File(sDir, "dMat.txt");
                if (matFile.exists() && secondaryLoadFlag) {
                    // If it already exists and the flag allows the load, load
                    // from the disk.
                    distMatrixSecondary = loadDMatFromFile(matFile);
                } else {
                    // Calculate the secondary distance matrix.
                    SharedNeighborFinder snf =
                            new SharedNeighborFinder(nsfPrimary);
                    snf.setNumClasses(numClasses);
                    snf.setPrimaryMetricsCalculator(primaryCMet);
                    float theta = 0;
                    snf.obtainWeightsFromHubnessInformation(theta);
                    snf.countSharedNeighborsMultiThread(8);
                    // As we initially obtain the similarities, we have to
                    // convert them to distances.
                    distMatrixSecondary = snf.getSharedNeighborCounts();
                    for (int i = 0; i < distMatrixSecondary.length; i++) {
                        for (int j = 0; j
                                < distMatrixSecondary[i].length; j++) {
                            distMatrixSecondary[i][j] =
                                    50 - distMatrixSecondary[i][j];
                        }
                    }
                    // Initialize the secondary CombinedMetric object.
                    secondaryCMet = new SharedNeighborCalculator(snf,
                            SharedNeighborCalculator.
                            WeightingType.HUBNESS_INFORMATION);

                }
                // Load or calculate the secondary kNN sets.
                File neighborsFile = new File(sDir, "knnSets.txt");
                if (neighborsFile.exists() && secondaryLoadFlag) {
                    nsfSecondary = NeighborSetFinder.loadNSF(neighborsFile,
                            quantizedRepresentation);
                    nsfSecondary.setDistances(distMatrixSecondary);
                } else {
                    nsfSecondary = new NeighborSetFinder(
                            quantizedRepresentation, distMatrixSecondary,
                            secondaryCMet);
                    nsfSecondary.calculateNeighborSetsMultiThr(50, 8);
                }
                printDMatToFile(distMatrixSecondary, matFile);
                nsfSecondary.saveNeighborSets(neighborsFile);
                neighborStatsCalculated = false;
                imageCoordinatesXY = null;
                JOptionPane.showMessageDialog(frameReference,
                        "Simhub calculated");
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * Renderer for the class distribution.
     */
    private class ClassDistrRenderer extends StackedBarRenderer {

        @Override
        public Paint getItemPaint(int row, int col) {
            return classColors[col];
        }
    }

    /**
     * A custom table cell renderer for the class-to-class hubness matrix that
     * is similar to the kNN confusion matrix.
     */
    class ClassToClassHubnessMatrixRenderer extends DefaultTableCellRenderer {

        private float[][] classToClassHubnessMatrix;
        private int numClasses;

        /**
         * Initialization.
         *
         * @param classToClassHubnessMatrix float[][] that is the class-to-class
         * hubness matrix that is similar to the kNN confusion matrix.
         * @param numClasses Integer that is the number of classes.
         */
        public ClassToClassHubnessMatrixRenderer(
                float[][] classToClassHubnessMatrix, int numClasses) {
            this.numClasses = numClasses;
            this.classToClassHubnessMatrix = classToClassHubnessMatrix;
        }

        @Override
        public Component getTableCellRendererComponent(
                JTable table,
                Object value,
                boolean isSelected,
                boolean hasFocus,
                int row,
                int column) {
            Component comp = super.getTableCellRendererComponent(
                    table, value, isSelected, hasFocus, row, column);
            // We treat the diagonal elements differently from the rest.
            if (row == column) {
                comp.setBackground(new java.awt.Color(50, 50
                        + (int) Math.max(0,
                        Math.min(205
                        * classToClassHubnessMatrix[row][column], 205)), 50));
            } else {
                comp.setBackground(new java.awt.Color(50
                        + (int) Math.max(0, Math.min(205
                        * classToClassHubnessMatrix[row][column], 205)),
                        50, 50));
            }
            return comp;
        }
    }

    /**
     * This class performs the data import from the workspace.
     */
    private class ImportHelper implements Runnable {

        /**
         * Default constructor.
         */
        public ImportHelper() {
        }

        @Override
        public void run() {
            if (busyCalculating) {
                // If the system is currently performing calculations, abort the
                // call.
                return;
            }
            busyCalculating = true;
            try {
                // Directory where the quantized representation is, if provided.
                File qRepDir = new File(workspace, "representation"
                        + File.separator + "quantized");
                // Get all the files in the directory.
                File[] reps = qRepDir.listFiles(
                        new util.fileFilters.ARFFFileNameFilter());
                if (reps.length > 0) {
                    IOARFF pers = new IOARFF();
                    BufferedReader br = null;
                    try {
                        // Only take the first representation, as there is no
                        // way to know which one to prefer in case of multiple
                        // files.
                        quantizedRepresentation = pers.load(reps[0].getPath());
                        numClasses = quantizedRepresentation.countCategories();
                        numClassesLabelValue.setText(
                                (new Integer(numClasses)).toString());
                        // Get the class colors.
                        ColorPalette paleta = new ColorPalette();
                        classColors = paleta.getClassColorArray(numClasses);
                        collectionSizeLabelValue.setText(
                                (new Integer(
                                quantizedRepresentation.size())).toString());
                        // Read the class names.
                        br = new BufferedReader(new InputStreamReader(
                                new FileInputStream(new File(workspace,
                                "classNames.txt"))));
                        String s = br.readLine();
                        classNames = s.split(",");
                        for (int c = 0; c < numClasses; c++) {
                            classColorAndNamesPanel.add(
                                    new ClassNameAndColorPanel(
                                    classNames[Math.min(
                                    c, classNames.length - 1)],
                                    classColors[c]));
                        }
                        // Class name vector for setting the column names in
                        // the class-to-class hubness matrix.
                        Vector cNameVector = new Vector(classNames.length);
                        for (int c = 0; c < numClasses; c++) {
                            cNameVector.add(classNames[c]);
                        }
                        TableModel classToClassModel =
                                new DefaultTableModel(numClasses, numClasses) {
                            @Override
                            public boolean isCellEditable(int row, int column) {
                                return false;
                            }
                        };
                        ((DefaultTableModel) classToClassModel).
                                setColumnIdentifiers(cNameVector);
                        classHubnessTable.setModel(classToClassModel);
                        DefaultCategoryDataset cDistDataset =
                                new DefaultCategoryDataset();
                        for (int c = 0; c < numClasses; c++) {
                            cDistDataset.addValue(
                                    quantizedRepresentation.
                                    getClassFrequencies()[c], "Size",
                                    classNames[c]);
                        }
                        for (int i = 0; i < numClasses; i++) {
                            classHubnessTable.getColumn(
                                    cNameVector.get(i)).setMinWidth(45);
                            classHubnessTable.getColumn(
                                    cNameVector.get(i)).setPreferredWidth(45);
                        }

                        classHubnessTable.setMinimumSize(
                                new Dimension(45 * numClasses,
                                45 * numClasses));
                        // Visualize the class distribution.
                        JFreeChart chart = ChartFactory.createBarChart3D(
                                "Class Distribution", "Category", "Size",
                                cDistDataset, PlotOrientation.VERTICAL,
                                false, true, false);
                        CategoryPlot plot = chart.getCategoryPlot();

                        plot.setRenderer(new ClassDistrRenderer());
                        ChartPanel chartPanel = new ChartPanel(chart);
                        chartPanel.setPreferredSize(new Dimension(440, 240));
                        classDistributionHolder.removeAll();
                        classDistributionHolder.add(chartPanel);
                        classDistributionHolder.revalidate();
                        classDistributionHolder.repaint();
                        classColorAndNamesPanel.validate();
                        classColorAndNamesPanel.repaint();
                        System.out.println("rep load successfull");
                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                    } finally {
                        if (br != null) {
                            br.close();
                        }
                    }
                } else {
                    System.out.println("No quantized representation detected.");
                    return;
                }
                // Index of the feature containing the image path information.
                int pathFeatureIndex = quantizedRepresentation.
                        getIndexForAttributeName("relative_path");
                File imageFile;
                // Load the thumbnails, a list of image paths, make a path map.
                images = new BufferedImage[quantizedRepresentation.size()];
                thumbnails = new ArrayList<>(quantizedRepresentation.size());
                imgPaths = new ArrayList<>(quantizedRepresentation.size());
                imgThumbPaths = new ArrayList<>(quantizedRepresentation.size());
                pathIndexMap =
                        new HashMap<>(4 * quantizedRepresentation.size());
                pathIndexMapThumbnail =
                        new HashMap<>(4 * quantizedRepresentation.size());
                for (int i = 0; i < quantizedRepresentation.size(); i++) {
                    imageFile = new File(workspace, "photos"
                            + quantizedRepresentation.getInstance(i).sAttr[
                            pathFeatureIndex]);
                    imgPaths.add(imageFile.getPath());
                    pathIndexMap.put(imageFile.getPath(), i);
                    imageFile = new File(workspace, "thumbnails"
                            + quantizedRepresentation.getInstance(i).sAttr[
                            pathFeatureIndex]);
                    imgThumbPaths.add(imageFile.getPath());
                    pathIndexMapThumbnail.put(imageFile.getPath(), i);
                    try {
                        thumbnails.add(ImageIO.read(imageFile));
                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                        System.out.println(imageFile.getPath());
                    }
                }
                JOptionPane.showMessageDialog(frameReference,
                        "Import completed");
            } catch (IOException | HeadlessException eSecond) {
                System.err.println(eSecond.getMessage());
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * Load distances and neighbor sets.
     *
     * @param matFile File that is the distance matrix file.
     * @param distanceModeIndicator Integer indicating whether to use primary or
     * secondary distance matrix load.
     */
    private void loadDistancesAndNeighbors(File matFile,
            int distanceModeIndicator) {
        File parDir = matFile.getParentFile();
        File neighborsFile = new File(parDir, "knnSets.txt");
        Thread t = new Thread(new DistancesNeighborsLoaderHelper(
                matFile, neighborsFile, distanceModeIndicator));
        t.start();
    }

    /**
     * This class loads the distances and neighbor sets from the files on the
     * disk.
     */
    private class DistancesNeighborsLoaderHelper implements Runnable {

        private File neighborsFile;
        private File matFile;
        private int distanceModeIndicator;

        /**
         * Initialization.
         *
         * @param matFile File that is the distance matrix file.
         * @param neighborsFile File that is the kNN file.
         * @param distanceModeIndicator Integer indicating whether this is a
         * primary or a secondary distance matrix load.
         */
        public DistancesNeighborsLoaderHelper(File matFile,
                File neighborsFile, int distanceModeIndicator) {
            this.matFile = matFile;
            this.neighborsFile = neighborsFile;
            this.distanceModeIndicator = distanceModeIndicator;
        }

        @Override
        public void run() {
            try {
                if (busyCalculating == true) {
                    // If the system is currently performing calculations,
                    // abort the call.
                    return;
                }
                busyCalculating = true;
                // Models are invalidated once the different kNN sets and
                // distances are loaded.
                trainedModels = false;
                if (distanceModeIndicator % 2 == PRIMARY_METRIC) {
                    // Primary matrix load.
                    if (matFile.exists()) {
                        distMatrixPrimary = loadDMatFromFile(matFile);
                    }
                    // Load or calculate the neighbors.
                    if (neighborsFile.exists()) {
                        nsfPrimary = NeighborSetFinder.loadNSF(
                                neighborsFile, quantizedRepresentation);
                        nsfPrimary.setDistances(distMatrixPrimary);
                    } else {
                        nsfPrimary = new NeighborSetFinder(
                                quantizedRepresentation, distMatrixPrimary,
                                primaryCMet);
                        nsfPrimary.calculateNeighborSetsMultiThr(50, 8);
                    }
                } else {
                    // Secondary distance load.
                    if (matFile.exists()) {
                        distMatrixSecondary = loadDMatFromFile(matFile);
                    }
                    // Load or calculate the neighbors.
                    if (neighborsFile.exists()) {
                        nsfSecondary = NeighborSetFinder.loadNSF(
                                neighborsFile, quantizedRepresentation);
                        nsfSecondary.setDistances(distMatrixSecondary);
                    } else {
                        if (secondaryCMet == null) {
                            secondaryCMet = new MutualProximityCalculator(
                                    distMatrixPrimary, quantizedRepresentation,
                                    primaryCMet);
                        }
                        nsfSecondary = new NeighborSetFinder(
                                quantizedRepresentation, distMatrixSecondary,
                                secondaryCMet);
                        nsfSecondary.calculateNeighborSetsMultiThr(50, 8);
                    }
                }
                rnnSetsAllK = new ArrayList[50][quantizedRepresentation.size()];

                // Update the occurrence profiles and the reverse neighbor
                // lists.
                int[][] kneighbors = getNSF().getKNeighbors();
                int size = quantizedRepresentation.size();
                occurrenceProfilesAllK = new float[50][
                        quantizedRepresentation.size()][numClasses];
                for (int j = 0; j < size; j++) {
                    rnnSetsAllK[0][j] = new ArrayList<>(2);
                }
                for (int j = 0; j < size; j++) {
                    rnnSetsAllK[0][kneighbors[j][0]].add(j);
                    occurrenceProfilesAllK[0][kneighbors[j][0]][
                            quantizedRepresentation.getLabelOf(j)]++;
                }
                // For all neighborhood sizes.
                for (int kTmp = 1; kTmp < 50; kTmp++) {
                    if (kneighbors[0].length >= kTmp + 1) {
                        for (int j = 0; j < size; j++) {
                            rnnSetsAllK[kTmp][j] =
                                    new ArrayList<>(2 * (kTmp + 1));
                            rnnSetsAllK[kTmp][j].addAll(
                                    rnnSetsAllK[kTmp - 1][j]);
                            occurrenceProfilesAllK[kTmp][j] =
                                    Arrays.copyOf(
                                    occurrenceProfilesAllK[kTmp - 1][j],
                                    numClasses);
                        }
                        for (int j = 0; j < size; j++) {
                            rnnSetsAllK[kTmp][kneighbors[j][kTmp]].add(j);
                            occurrenceProfilesAllK[kTmp][kneighbors[j][kTmp]][
                                    quantizedRepresentation.getLabelOf(j)]++;
                        }
                    } else {
                        for (int j = 0; j < size; j++) {
                            rnnSetsAllK[kTmp][j] = new ArrayList<>(1);
                        }
                    }
                }
                // Reinitialize.
                aboveZeroArray = null;
                skewArray = null;
                kurtosisArray = null;
                highestHubnesses = null;
                highestHubIndexes = null;
                kEntropies = null;
                reverseKNNEntropies = null;
                kEntropySkews = null;
                reverseKNNEntropySkews = null;
                badHubnessArray = null;
                globalClassToClasshubness = null;
                hubPercs = null;
                orphanPercs = null;
                regularPercs = null;
                graphsDelete();
                for (int c1 = 0; c1 < numClasses; c1++) {
                    for (int c2 = 0; c2 < numClasses; c2++) {
                        classHubnessTable.setValueAt(" ", c1, c2);
                    }
                }
                percAboveLabelValue.setText("...");
                skewnessLabelValue.setText("...");
                kurtosisLabelValue.setText("...");
                majorDegLabelValue.setText("...");
                nkEntropySkewnessValues.setText("...");
                nkEntropyLabelValue.setText("...");
                rnkEntropySkewnessValue.setText("...");
                rnkEntropyValue.setText("...");
                badHubnessLabelValue.setText("...");
                imageCoordinatesXY = null;
                JOptionPane.showMessageDialog(frameReference,
                        "Distances properly loaded.");
            } catch (Exception e) {
                System.err.println(e.getMessage());
                JOptionPane.showMessageDialog(frameReference,
                        "An error occurred: " + e.getMessage(),
                        "Error message", JOptionPane.ERROR_MESSAGE);
            } finally {
                busyCalculating = false;
            }
        }
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(ImageHubExplorer.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(ImageHubExplorer.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(ImageHubExplorer.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(ImageHubExplorer.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                new ImageHubExplorer().setVisible(true);
            }
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton addNNsButton;
    private javax.swing.JButton addRNNsButton;
    private javax.swing.JButton addSelectedButton;
    private javax.swing.JLabel badHubnessLabelTxt;
    private javax.swing.JLabel badHubnessLabelValue;
    private javax.swing.JMenuItem bcMenuItem;
    private javax.swing.JScrollPane cNamesScrollPane;
    private javax.swing.JMenuItem canMenuItem;
    private javax.swing.JPanel chartHoldingPanelOccDistribution;
    private javax.swing.JPanel classColorAndNamesPanel;
    private javax.swing.JPanel classDistributionHolder;
    private javax.swing.JTable classHubnessTable;
    private javax.swing.JPanel classPanel;
    private javax.swing.JScrollPane classesScrollPane;
    private javax.swing.JPanel classesScrollPanel;
    private javax.swing.JMenu classificationMenu;
    private javax.swing.JPanel classifierPredictionsPanel;
    private javax.swing.JMenu codebookMenu;
    private javax.swing.JMenu collectionMenu;
    private javax.swing.JButton collectionSearchButton;
    private javax.swing.JLabel collectionSizeLabelTxt;
    private javax.swing.JLabel collectionSizeLabelValue;
    private javax.swing.JScrollPane confusionMatScrollPane;
    private javax.swing.JMenu dMatrixMenu;
    private javax.swing.JPanel dataMainPanel;
    private javax.swing.JMenuItem distCalcCosineItem;
    private javax.swing.JMenuItem distCalcEuclideanItem;
    private javax.swing.JMenu distCalculateMenu;
    private javax.swing.JMenuItem distImportItem;
    private javax.swing.JMenu editMenu;
    private javax.swing.JMenuItem graphScreenCaptureItem;
    private javax.swing.JLabel hRelatedPropTxt;
    private javax.swing.JTabbedPane hubTab;
    private javax.swing.JLabel hubsLabelTxt;
    private javax.swing.JLabel hubsLabelValue;
    private javax.swing.JButton imageBrowseButton;
    private javax.swing.JMenuItem importItem;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JScrollPane jScrollPane3;
    private javax.swing.JScrollPane jScrollPane4;
    private javax.swing.JScrollPane jScrollPane5;
    private javax.swing.JTextField jTextField1;
    private javax.swing.JSlider kSelectionSlider;
    private javax.swing.JMenuItem klMenuItem;
    private javax.swing.JLabel kurtosisLabelTxt;
    private javax.swing.JLabel kurtosisLabelValue;
    private javax.swing.JMenuItem loadCodebookItem;
    private javax.swing.JMenuItem loadCodebookProfileMenuItem;
    private javax.swing.JMenuItem loadSecondaryDistancesItem;
    private javax.swing.JMenuItem localScalingItem;
    private javax.swing.JLabel majorDegLabelTxt;
    private javax.swing.JLabel majorDegLabelValue;
    private javax.swing.JMenuItem majorHubSelectionItem;
    private javax.swing.JMenuItem manhattanDistItem;
    private gui.images.ImagesDisplayPanel mdsCollectionPanel;
    private javax.swing.JMenuItem mdsScreenCaptureItem;
    private javax.swing.JScrollPane mdsScrollPane;
    private javax.swing.JMenuItem mdsVisualizeItem;
    private javax.swing.JMenuBar menuBar;
    private javax.swing.JMenu metricLearningMenu;
    private javax.swing.JMenuItem mpMenuItem;
    private javax.swing.JScrollPane neighborGraphScrollPane;
    private javax.swing.JPanel neighborPanel;
    private javax.swing.JMenuItem neighborStatsItem;
    private javax.swing.JMenuItem nextMenuItem;
    private javax.swing.JLabel nhSizeLabelTxt;
    private javax.swing.JMenuItem nicdmItem;
    private javax.swing.JLabel nkEntropyLabelTxt;
    private javax.swing.JLabel nkEntropyLabelValue;
    private javax.swing.JLabel nkEntropySkewnessTxt;
    private javax.swing.JLabel nkEntropySkewnessValues;
    private javax.swing.JPanel nnPanel;
    private javax.swing.JLabel nnScrollLabelTxt;
    private javax.swing.JScrollPane nnScrollPane;
    private javax.swing.JLabel noccProfLabelTxt;
    private javax.swing.JLabel numClassesLabelTxt;
    private javax.swing.JLabel numClassesLabelValue;
    private javax.swing.JPanel occProfileChartHolder;
    private javax.swing.JLabel orphansLabelTxt;
    private javax.swing.JLabel orphansLabelValue;
    private javax.swing.JLabel percAboveLabelTxt;
    private javax.swing.JLabel percAboveLabelValue;
    private javax.swing.JLabel prClassLabelTxt;
    private javax.swing.JScrollPane prClassScrollPane;
    private javax.swing.JMenuItem previousMenuItem;
    private gui.images.ImagePanel queryImagePanel;
    private javax.swing.JPanel queryNNPanel;
    private javax.swing.JScrollPane queryNNScrollPane;
    private javax.swing.JLabel queryQTextLabelTxt;
    private javax.swing.JButton reRankingButton;
    private javax.swing.JLabel regularLabelTxt;
    private javax.swing.JLabel regularLabelValue;
    private javax.swing.JButton removeAllButton;
    private javax.swing.JButton removeVertexButton;
    private javax.swing.JLabel rnkEntropySkewnessTxt;
    private javax.swing.JLabel rnkEntropySkewnessValue;
    private javax.swing.JLabel rnkEntropyValue;
    private javax.swing.JPanel rnnPanel;
    private javax.swing.JLabel rnnScrollLabelTxt;
    private javax.swing.JScrollPane rnnScrollPane;
    private javax.swing.JMenu screenCaptureMenu;
    private javax.swing.JButton searchButton;
    private javax.swing.JPanel searchPanel;
    private javax.swing.JLabel searchQLabelTxt;
    private javax.swing.JMenu secondaryMetricMenu;
    private javax.swing.JMenu selImageMenu;
    private javax.swing.JMenuItem selImgPathMenuItem;
    private javax.swing.JMenuItem selSIFTmenuItem;
    private javax.swing.JLabel selectedImageLabelClass;
    private javax.swing.JLabel selectedImageLabelClassNeighbor;
    private javax.swing.JLabel selectedImageLabelClassNeighborMain;
    private javax.swing.JLabel selectedImageLabelSearch;
    private gui.images.ImagePanel selectedImagePanelClass;
    private gui.images.ImagePanel selectedImagePanelClassNeighbor;
    private gui.images.ImagePanel selectedImagePanelClassNeighborMain;
    private gui.images.ImagePanel selectedImagePanelSearch;
    private javax.swing.JLabel selectedImagePathLabelClass;
    private javax.swing.JLabel selectedImagePathLabelClassNeighbor;
    private javax.swing.JLabel selectedImagePathLabelClassNeighborMain;
    private javax.swing.JLabel selectedImagePathLabelSearch;
    private javax.swing.JLabel simResLabelTxt;
    private javax.swing.JMenuItem simcosMenuItem;
    private javax.swing.JMenuItem simhubMenuItem;
    private javax.swing.JLabel skewnessLabelValue;
    private javax.swing.JLabel skewnwessLabelTxt;
    private javax.swing.JMenuItem tanimotoMenuItem;
    private javax.swing.JMenuItem trainModelsItem;
    private javax.swing.JLabel workspaceLabelTxt;
    private javax.swing.JLabel workspaceLabelValue;
    private javax.swing.JMenuItem workspaceMenuItem;
    // End of variables declaration//GEN-END:variables
}
