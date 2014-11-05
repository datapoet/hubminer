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
package images.mining.clustering;

import data.representation.DataSet;
import data.representation.images.sift.LFeatRepresentation;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import images.mining.display.SIFTDraw;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.images.SiftUtil;
import java.awt.image.BufferedImage;
import java.io.File;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.evaluation.quality.OptimalConfigurationFinder;
import learning.unsupervised.methods.KMeans;
import util.CommandLineParser;

/**
 * This utility class enables the users to cluster the SIFT features on a single
 * image and output the clusters in an ARFF format.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SingleImageSIFTClusterer {

    private LFeatRepresentation features = null;
    private CombinedMetric cmet = null;
    private OptimalConfigurationFinder selector = null;
    private int minClusters;
    private int maxClusters;
    private static final int NUM_REPETITIONS = 10;

    /**
     * @return SIFTRepresentation object that holds the features on the image.
     */
    public LFeatRepresentation getFeatures() {
        return features;
    }

    /**
     * @param minClusters Integer that is the minimal number of clusters to try
     * and use.
     * @param maxClusters Integer that is the maximal number of clusters to try
     * and use.
     */
    public void setBounds(int minClusters, int maxClusters) {
        this.minClusters = minClusters;
        this.maxClusters = maxClusters;
    }

    /**
     * @param cmet CombinedMetric object for distance calculations.
     */
    public void setCombinedMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @param features SIFTRepresentation object that holds the features on the
     * image.
     */
    public void setRepresentation(LFeatRepresentation features) {
        this.features = features;
    }

    /**
     * @param selector OptimalConfigurationFinder for finding the optimal
     * cluster configuration among all generated configurations.
     */
    public void setConfigurationSelector(OptimalConfigurationFinder selector) {
        this.selector = selector;
    }

    /**
     * Loads the SIFT features of an image from an ARFF file.
     *
     * @param inPath String that is the path to the ARFF file containing the
     * image SIFT features.
     */
    private void loadFeatures(String inPath) {
        try {
            features = SiftUtil.importFeaturesFromArff(inPath);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     */
    public SingleImageSIFTClusterer() {
    }

    /**
     * Initialization.
     *
     * @param inPath String that is the path to the ARFF file containing the
     * image SIFT features.
     * @param cmet CombinedMetric object for distance calculations.
     * @param selector OptimalConfigurationFinder for finding the optimal
     * cluster configuration among all generated configurations.
     * @param minClusters Integer that is the minimal number of clusters to try
     * and use.
     * @param maxClusters Integer that is the maximal number of clusters to try
     * and use.
     */
    public SingleImageSIFTClusterer(String inPath, CombinedMetric cmet,
            OptimalConfigurationFinder selector, int minClusters,
            int maxClusters) {
        loadFeatures(inPath);
        this.selector = selector;
        this.cmet = cmet;
        this.minClusters = minClusters;
        this.maxClusters = maxClusters;
    }

    /**
     * Perform a single clustering run for a fixed number of clusters on the
     * features loaded from an ARFF file and persist the clusters in an output
     * ARFF file.
     *
     * @param inPath String that is the path to the ARFF file containing the
     * image SIFT features.
     * @param outPath
     * @param numClust
     * @throws Exception
     */
    public static void clusterARFFToARFF(String inPath,
            String outPath, int numClust) throws Exception {
        File inFile = new File(inPath);
        if (numClust < 1) {
            throw new Exception("Number of clusters must be positive, not "
                    + numClust);
        }
        if (!inFile.exists() || !inFile.isFile()) {
            throw new Exception("Bad input path " + inPath);
        }
        Cluster[] clusterConfiguration;
        // Load the features.
        LFeatRepresentation features = SiftUtil.importFeaturesFromArff(inPath);
        CombinedMetric cmet = new CombinedMetric(null,
                new MinkowskiMetric(), CombinedMetric.DEFAULT);
        // Perform the clustering.
        KMeans clusterer = new KMeans(features, cmet, numClust);
        clusterer.cluster();
        clusterConfiguration = clusterer.getClusters();
        DataSet outDSet = new DataSet();
        outDSet.fAttrNames =
                clusterConfiguration[0].getDefinitionDataset().fAttrNames.
                clone();
        outDSet.iAttrNames = new String[1];
        outDSet.iAttrNames[0] = "Cluster";
        outDSet.sAttrNames = new String[1];
        outDSet.sAttrNames[0] = "ImageName";
        persistClusters(outPath, clusterConfiguration, outDSet);
    }

    /**
     * This method performs the clustering and finds the optimal cluster
     * configuration based on a product of different cluster validity indices.
     *
     * @return Cluster[] that is the cluster configuration produced by the
     * clustering and selected as the optimal one.
     * @throws Exception
     */
    public Cluster[] clusterImageByIndexProduct() throws Exception {
        Cluster[][] clusteringConfigurations =
                new Cluster[(maxClusters - minClusters + 1) *
                NUM_REPETITIONS][];
        Cluster[][] nonDuplicateConfigurationArray =
                new Cluster[(maxClusters - minClusters + 1)][];
        Cluster[][] repetitionsArray;
        selector.setDataSet(features);
        for (int nClusters = minClusters; nClusters <= maxClusters;
                nClusters++) {
            System.out.println("Clustering for : " + nClusters + " clusters.");
            repetitionsArray = new Cluster[NUM_REPETITIONS][];
            for (int rIndex = 0; rIndex < NUM_REPETITIONS; rIndex++) {
                KMeans clusterer = new KMeans(features, cmet, nClusters);
                clusterer.cluster();
                clusteringConfigurations[NUM_REPETITIONS
                        * (nClusters - minClusters) + rIndex] =
                        clusterer.getClusters();
                repetitionsArray[rIndex] = clusteringConfigurations[
                        NUM_REPETITIONS * (nClusters - minClusters) + rIndex];
            }
            selector.setConfigurationList(repetitionsArray);
            nonDuplicateConfigurationArray[nClusters - minClusters] =
                    selector.findBestConfigurationByIndexProduct();
        }
        selector.setConfigurationList(nonDuplicateConfigurationArray);
        Cluster[] bestConfig = selector.findBestConfigurationByIndexProduct();
        return bestConfig;
    }

    /**
     * Performs the clustering across the specified cluster range of the SIFT
     * features of an image and finds the optimal cluster configuration
     * according to all supported cluster validity indices. It outputs all such
     * optimal configurations.
     *
     * @return Cluster[][] containing the optimal reached cluster configurations
     * of the SIFT features of an image, for all supported cluster validity
     * indices.
     * @throws Exception
     */
    public Cluster[][] clusterImageAll() throws Exception {
        Cluster[][] chosenConfigurations =
                new Cluster[OptimalConfigurationFinder.NUM_INDEXES][];
        Cluster[][] clusteringConfigurations =
                new Cluster[(maxClusters - minClusters + 1) * NUM_REPETITIONS][];
        Cluster[][][] nonDuplicateConfigurationArray = new Cluster[
                OptimalConfigurationFinder.NUM_INDEXES][
                (maxClusters - minClusters + 1)][];
        Cluster[][] repetitionsArray;
        selector.setDataSet(features);
        for (int nClusters = minClusters; nClusters <= maxClusters;
                nClusters++) {
            System.out.println("Clustering for : " + nClusters + " clusters.");
            repetitionsArray = new Cluster[NUM_REPETITIONS][];
            for (int rIndex = 0; rIndex < NUM_REPETITIONS; rIndex++) {
                KMeans clusterer = new KMeans(features, cmet, nClusters);
                clusterer.cluster();
                clusteringConfigurations[NUM_REPETITIONS
                        * (nClusters - minClusters) + rIndex] =
                        clusterer.getClusters();
                repetitionsArray[rIndex] = clusteringConfigurations[
                        NUM_REPETITIONS * (nClusters - minClusters) + rIndex];
            }
            selector.setConfigurationList(repetitionsArray);
            for (int j = 0; j < OptimalConfigurationFinder.NUM_INDEXES; j++) {
                selector.setConfigurationList(repetitionsArray);
                selector.setQualityIndex(j);
                nonDuplicateConfigurationArray[j][nClusters - minClusters] =
                        selector.findBestConfiguration();
            }
        }
        for (int j = 0; j < OptimalConfigurationFinder.NUM_INDEXES; j++) {
            selector.setConfigurationList(nonDuplicateConfigurationArray[j]);
            selector.setQualityIndex(j);
            chosenConfigurations[j] = selector.findBestConfiguration();
        }
        return chosenConfigurations;
    }

    /**
     * Perform clustering of the SIFT features on an image.
     *
     * @param bi BufferedImage object that is the image where the SIFT features
     * are being clustered. It is used in the adapted K-means, if the 'simple'
     * flag is set to false, for extracting color information in neighborhoods
     * of the keypoints.
     * @param simple Boolean flag indicating whether the basic K-means is used
     * or the adapted version.
     * @return Cluster[] that is the SIFT feature cluster configuration that is
     * found to be optimal.
     * @throws Exception
     */
    public Cluster[] clusterImage(BufferedImage bi,
            boolean simple) throws Exception {
        Cluster[][] clusteringConfigurations =
                new Cluster[(maxClusters - minClusters + 1) *
                NUM_REPETITIONS][];
        Cluster[][] nonDuplicateConfigurationArray =
                new Cluster[(maxClusters - minClusters + 1)][];
        Cluster[][] repetitionsArray;
        selector.setDataSet(features);
        scaleDownDescriptor(3.7f);
        for (int nClusters = minClusters; nClusters <= maxClusters;
                nClusters++) {
            System.out.println("Clustering for : " + nClusters + " clusters.");
            repetitionsArray = new Cluster[NUM_REPETITIONS][];
            for (int rIndex = 0; rIndex < NUM_REPETITIONS; rIndex++) {
                KMeans km = new KMeans(features, cmet, nClusters);
                IntraImageKMeansAdapted ikma =
                        new IntraImageKMeansAdapted(bi, features, nClusters);
                ClusteringAlg clusterer = simple ? km : ikma;
                clusterer.cluster();
                clusteringConfigurations[NUM_REPETITIONS
                        * (nClusters - minClusters) + rIndex] =
                        clusterer.getClusters();
                repetitionsArray[rIndex] = clusteringConfigurations[
                        NUM_REPETITIONS * (nClusters - minClusters) + rIndex];
            }
            selector.setConfigurationList(repetitionsArray);
            nonDuplicateConfigurationArray[nClusters - minClusters] =
                    selector.findBestConfiguration();
        }
        selector.setConfigurationList(nonDuplicateConfigurationArray);
        Cluster[] bestConfig = selector.findBestConfiguration();
        return bestConfig;
    }

    /**
     * Persist the clusters in a specified ARFF file and at the same time fill a
     * DataSet object from the Cluster[] cluster configuration array.
     *
     * @param outPath String that is the path to the output file for the cluster
     * configuration.
     * @param clusterConfiguration Cluster[] that is the cluster configuration
     * to persist.
     * @param outDSet DataSet object to fill with the features from the cluster
     * configuration.
     * @throws Exception
     */
    public static void persistClusters(String outPath,
            Cluster[] clusterConfiguration, DataSet outDSet) throws Exception {
        if ((clusterConfiguration != null)
                && (clusterConfiguration.length > 0)) {
            File outFile = new File(outPath);
            FileUtil.createFile(outFile);
            for (int cIndex = 0; cIndex < clusterConfiguration.length;
                    cIndex++) {
                for (int j = 0; j < clusterConfiguration[cIndex].size(); j++) {
                    outDSet.data.add(
                            clusterConfiguration[cIndex].getInstance(j));
                    clusterConfiguration[cIndex].getInstance(j).iAttr =
                            new int[1];
                    clusterConfiguration[cIndex].getInstance(j).iAttr[0] =
                            cIndex;
                    clusterConfiguration[cIndex].getInstance(j).
                            embedInDataset(outDSet);
                }
            }
            IOARFF persister = new IOARFF();
            persister.save(outDSet, outPath, null);
        }
    }

    /**
     *
     * @param outPath
     * @param clusterConfiguration
     * @throws Exception
     */
    public static void persistClusters(String outPath,
            Cluster[] clusterConfiguration) throws Exception {
        if ((clusterConfiguration != null)
                && (clusterConfiguration.length > 0)) {
            File outFile = new File(outPath);
            FileUtil.createFile(outFile);
            DataSet outDSet = new DataSet();
            outDSet.fAttrNames =
                    clusterConfiguration[0].getDefinitionDataset().fAttrNames;
            outDSet.sAttrNames =
                    clusterConfiguration[0].getDefinitionDataset().sAttrNames;
            // As SIFT features don't have categories, this way of handling it
            // is not entirely necessary, instance labels could be used instead.
            outDSet.iAttrNames = new String[1];
            outDSet.iAttrNames[0] = "Cluster";
            outDSet.identifiers = clusterConfiguration[0].
                    getDefinitionDataset().identifiers;
            for (int cIndex = 0; cIndex < clusterConfiguration.length;
                    cIndex++) {
                for (int j = 0; j < clusterConfiguration[cIndex].size(); j++) {
                    outDSet.data.add(
                            clusterConfiguration[cIndex].getInstance(j));
                    clusterConfiguration[cIndex].getInstance(j).iAttr =
                            new int[1];
                    clusterConfiguration[cIndex].getInstance(j).iAttr[0] =
                            cIndex;
                    clusterConfiguration[cIndex].getInstance(j).
                            embedInDataset(outDSet);
                }
            }
            IOARFF persister = new IOARFF();
            persister.save(outDSet, outPath, null);
        }
    }

    /**
     * Recursively intra-cluster SIFT features in a directory of images.
     *
     * @param inPath Current input path to the features.
     * @param outPath Current output path for the cluster configurations.
     * @param imagePath Current image path.
     * @throws Exception
     */
    public static void clusterImageDirectory(String inPath, String outPath,
            String imagePath) throws Exception {
        File inFile = new File(inPath);
        File outFile = new File(outPath);
        if (inFile.isDirectory()) {
            File[] children = inFile.listFiles();
            if (children != null) {
                for (int i = 0; i < children.length; i++) {
                    if (children[i].isFile()) {
                        clusterImageFileByIndexProduct(children[i].getPath(),
                                outPath + File.separator
                                + children[i].getName().substring(0,
                                children[i].getName().length() - 5),
                                imagePath + File.separator
                                + children[i].getName().substring(0,
                                children[i].getName().length() - 4) + "jpg");
                    } else {
                        clusterImageDirectory(children[i].getPath(),
                                outPath + File.separator
                                + children[i].getName(), imagePath
                                + File.separator + children[i].getName());
                    }
                }
            }
        } else if (outFile.isDirectory()) {
            clusterImageFileAll(inPath, outPath, imagePath);
        }
    }

    /**
     * When the adaptive clustering approach is used, it is prudent to scale
     * down the descriptor a bit, because of the way the weights are set up and
     * the preferences between proximity-based and descriptor-based
     * similarities.
     *
     * @param scaleFactor Float value that is the scale factor.
     * @throws Exception
     */
    private void scaleDownDescriptor(float scaleFactor) throws Exception {
        for (int i = 0; i < features.data.size(); i++) {
            // The first 4 values are the X,Y coordinates, scale and angle.
            for (int j = 4; j < features.fAttrNames.length; j++) {
                features.data.get(i).fAttr[j] /= scaleFactor;
            }
        }
    }

    /**
     * Cluster the SIFT features in an image and use a product of cluster
     * validity indices to decide on the optimal configuration. The clustered
     * features are drawn on the images and this modified copy is also persisted
     * to the output directory.
     *
     * @param inPath String that is path to the representational file.
     * @param outPath String that is the path to where to persist the output.
     * @param imagePath String that is the path to the image file.
     * @throws Exception
     */
    private static void clusterImageFileByIndexProduct(String inPath,
            String outPath, String imagePath) throws Exception {
        int minClusters = 10;
        int maxClusters = 30;
        CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;
        SingleImageSIFTClusterer imageClusterer;
        OptimalConfigurationFinder selector;
        Cluster[] clusterConfiguration;
        SIFTDraw drawUtil;
        selector = new OptimalConfigurationFinder();
        imageClusterer = new SingleImageSIFTClusterer(inPath, cmet, selector,
                minClusters, maxClusters);
        imageClusterer.scaleDownDescriptor(80f);
        clusterConfiguration = imageClusterer.clusterImageByIndexProduct();
        // Persist the clusters.
        persistClusters(outPath + File.separator + "best_config_product.arff",
                clusterConfiguration);
        drawUtil = new SIFTDraw(clusterConfiguration, imagePath);
        // Persist the image that the clusters of SIFT features are drawn onto.
        drawUtil.drawClusters(outPath + File.separator
                + "image_clusters_product.jpg");
    }

    /**
     * Cluster the SIFT features in an image and use a series of cluster
     * validity indices to decide on the optimal configurations. The clustered
     * features are drawn on the images and this modified copy is also persisted
     * to the output directory.
     *
     * @param inPath String that is path to the representational file.
     * @param outPath String that is the path to where to persist the output.
     * @param imagePath String that is the path to the image file.
     * @throws Exception
     */
    private static void clusterImageFileAll(String inPath, String outPath,
            String imagePath) throws Exception {
        int minClusters = 10;
        int maxClusters = 30;
        CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;
        SingleImageSIFTClusterer imageClusterer;
        OptimalConfigurationFinder selector;
        Cluster[][] clusterConfiguration;
        SIFTDraw drawUtil;
        selector = new OptimalConfigurationFinder();
        imageClusterer = new SingleImageSIFTClusterer(inPath, cmet, selector,
                minClusters, maxClusters);
        imageClusterer.scaleDownDescriptor(80f);
        // Persist the optimal cluster configuration according to each of the
        // used cluster validity indices.
        clusterConfiguration = imageClusterer.clusterImageAll();
        persistClusters(outPath + File.separator + "best_config_dunn.arff",
                clusterConfiguration[0]);
        persistClusters(outPath + File.separator + "best_config_db.arff",
                clusterConfiguration[1]);
        persistClusters(outPath + File.separator + "best_config_rs.arff",
                clusterConfiguration[2]);
        persistClusters(outPath + File.separator + "best_config_sd.arff",
                clusterConfiguration[3]);
        persistClusters(outPath + File.separator
                + "best_config_silhouette.arff", clusterConfiguration[4]);
        // Persist the drawn SIFT clusters on images.
        drawUtil = new SIFTDraw(clusterConfiguration[0], imagePath);
        drawUtil.drawClusters(outPath + File.separator
                + "image_clusters_dunn.jpg");
        drawUtil = new SIFTDraw(clusterConfiguration[1], imagePath);
        drawUtil.drawClusters(outPath + File.separator
                + "image_clusters_db.jpg");
        drawUtil = new SIFTDraw(clusterConfiguration[2], imagePath);
        drawUtil.drawClusters(outPath + File.separator
                + "image_clusters_rs.jpg");
        drawUtil = new SIFTDraw(clusterConfiguration[3], imagePath);
        drawUtil.drawClusters(outPath + File.separator
                + "image_clusters_sd.jpg");
        drawUtil = new SIFTDraw(clusterConfiguration[4], imagePath);
        drawUtil.drawClusters(outPath + File.separator
                + "image_clusters_silhouette.jpg");
    }

    /**
     * Performs intra-image SIFT feature clustering on a directory.
     *
     * @param args Command line parameters.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-arffInputDir", "Path to the input folder containing the "
                + "image SIFT representation ARFF-s.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-imgInputDir", "Path to the input folder containing the "
                + "image files.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outDir", "Path to the output directory",
                CommandLineParser.STRING, true, false);

        clp.parseLine(args);
        String inRepPath = (String) clp.getParamValues("-arffInputDir").get(0);
        String inImgPath = (String) clp.getParamValues("-imgInputDir").get(0);
        String outDirPath = (String) clp.getParamValues("-outDir").get(0);
        clusterImageDirectory(inRepPath, outDirPath, inImgPath);
    }
}