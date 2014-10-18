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
package images.mining.codebook;

import java.io.File;
import java.util.ArrayList;
import data.representation.images.sift.SIFTRepresentation;
import data.representation.images.sift.SIFTVector;
import ioformat.images.SiftUtil;
import util.fileFilters.ARFFFileNameFilter;
import util.fileFilters.KeyFileNameFilter;
import util.fileFilters.DirectoryFilter;
import ioformat.IOARFF;
import data.representation.DataInstance;
import learning.unsupervised.Cluster;
import distances.primary.SIFTMetric;
import distances.primary.CombinedMetric;
import learning.unsupervised.methods.FastKMeans;
import learning.unsupervised.ClusteringAlg;
import java.util.Random;

/**
 * This class implements the logic for calculating codebook vectors for SIFT
 * features.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTCodebookMaker {

    public static final int DEFAULT_SIZE = 400;
    private File target = null;
    private boolean recursive = true;
    private SIFTRepresentation siftSample = null;
    private Cluster[] resultingConfiguration = null;
    private FastKMeans clusterer = null;

    /**
     * The main method which takes exactly four command line parameters: input
     * file, output file, the dimensionality of the codebook to be generated and
     * the percentage of the sample to load.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.out.println("4 args");
            System.out.println("arg0: inPath to arff with "
                    + "features for clustering");
            System.out.println("arg1: outPath to file where codebooks are to "
                    + "be persisted");
            System.out.println("arg2: dimensionality of codebook");
            System.out.println("arg3: percentage of the features to load,"
                    + " as a float value in the (0, 1] range.");
        } else {
            SIFTCodebookMaker scm = new SIFTCodebookMaker(new File(args[0]),
                    false);
            scm.getSIFTFromTargetARFFs(Float.parseFloat(args[3]));
            scm.clusterFeatures(Integer.parseInt(args[2]));
            SIFTCodeBook generatedCodebook = scm.getCodeBookVectors();
            generatedCodebook.writeCodeBookToFile(new File(args[1]));
        }
    }

    public SIFTCodebookMaker(File target, boolean recursive) {
        // If the target is a directory, clustering will be done for all arff
        // files in the directory. If it is a file, only for the given file.
        this.target = target;
        this.recursive = recursive;
        siftSample = new SIFTRepresentation(20000, 10000);
    }

    /**
     * @return ClusteringAlg object used in quantization.
     */
    public ClusteringAlg getClusteringObject() {
        return clusterer;
    }

    /**
     * Perform clustering on the SIFT sample.
     *
     * @throws Exception
     */
    public void clusterFeatures() throws Exception {
        clusterFeatures(DEFAULT_SIZE);
    }

    /**
     * Perform clustering on the SIFT sample.
     *
     * @param numClusters Integer that is the number of clusters to generate.
     * @throws Exception
     */
    public void clusterFeatures(int numClusters) throws Exception {
        if ((siftSample == null) || (siftSample.isEmpty())) {
            return;
        }
        CombinedMetric cmet = new CombinedMetric(null, new SIFTMetric(),
                CombinedMetric.DEFAULT);
        clusterer = new FastKMeans(siftSample, cmet, numClusters);
        clusterer.cluster();
        resultingConfiguration = clusterer.getClusters();
    }

    /**
     * Get the generated cluster configuration.
     *
     * @return
     */
    public Cluster[] getConfiguration() {
        return resultingConfiguration;
    }

    /**
     * @return SIFTVector array of the generated clustering configuration
     * centroids.
     * @throws Exception
     */
    public SIFTVector[] getConfigurationCentroids() throws Exception {
        if ((resultingConfiguration == null)
                || resultingConfiguration.length == 0) {
            return new SIFTVector[0];
        }
        SIFTVector[] centroids = new SIFTVector[resultingConfiguration.length];
        for (int i = 0; i < centroids.length; i++) {
            centroids[i] = new SIFTVector(
                    resultingConfiguration[i].getCentroid());
        }
        return centroids;
    }

    /**
     * @return SIFTCodeBook object that corresponds to the generated centroids.
     * @throws Exception
     */
    public SIFTCodeBook getCodeBookVectors() throws Exception {
        if ((resultingConfiguration == null)
                || resultingConfiguration.length == 0) {
            return new SIFTCodeBook();
        }
        SIFTCodeBook result = new SIFTCodeBook();
        ArrayList<SIFTVector> featureVector = new ArrayList<>(
                resultingConfiguration.length);
        for (int i = 0; i < resultingConfiguration.length; i++) {
            featureVector.add(new SIFTVector(
                    resultingConfiguration[i].getCentroid()));
        }
        result.setCodeBookSet(featureVector);
        return result;
    }

    /**
     * Loads the features from a file in arff format.
     *
     * @param inSIFTFile Input file.
     */
    private void loadSIFTFromARFFFile(File inSIFTFile) {
        // Extract data and append to the current sample.
        IOARFF persister = new IOARFF();
        SIFTRepresentation imageSIFTrep;
        try {
            imageSIFTrep = new SIFTRepresentation(
                    persister.load(inSIFTFile.getPath()));
            if (!imageSIFTrep.isEmpty()) {
                for (DataInstance localFeature : imageSIFTrep.data) {
                    siftSample.addDataInstance(localFeature);
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * Loads the features from a file in Lowe's key format.
     *
     * @param inSIFTFile Input file.
     */
    private void loadSIFTFromSIFTFile(File inSIFTFile) {
        // Extract data and append to the current sample.
        SIFTRepresentation imageSIFTrep;
        try {
            imageSIFTrep = SiftUtil.importFeaturesFromSift(inSIFTFile);
            if (!imageSIFTrep.isEmpty()) {
                for (DataInstance element : imageSIFTrep.data) {
                    siftSample.addDataInstance(element);
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * Loads the sample from a directory of files.
     *
     * @param inDir Input directory.
     */
    private void loadSIFTFromDirectory(File inDir) {
        File[] arffFiles = inDir.listFiles(new ARFFFileNameFilter());
        File[] keyFiles = inDir.listFiles(new KeyFileNameFilter());
        File[] subdirectories = inDir.listFiles(new DirectoryFilter());
        if (recursive) {
            for (File dir : subdirectories) {
                loadSIFTFromDirectory(dir);
            }
        }
        for (File arffFile : arffFiles) {
            loadSIFTFromARFFFile(arffFile);
        }
        for (File keyFile : keyFiles) {
            loadSIFTFromSIFTFile(keyFile);
        }
    }

    /**
     * Loads the features from the current target.
     */
    public void getSIFTFromTargetARFFs() {
        if (target.isDirectory()) {
            loadSIFTFromDirectory(target);
        } else {
            loadSIFTFromARFFFile(target);
        }
    }

    /**
     * Loads the sample from a file.
     *
     * @param inFile Input file in arff format.
     * @param perc Float value that is the percentage of features to take.
     */
    private void loadSIFTFromARFFFile(File inFile, float perc) {
        // Extract data and append to the current sample.
        IOARFF persister = new IOARFF();
        SIFTRepresentation imageSIFTrep;
        Random randa = new Random();
        try {
            imageSIFTrep = new SIFTRepresentation(persister.load(
                    inFile.getPath()));
            if (!imageSIFTrep.isEmpty()) {
                for (DataInstance element : imageSIFTrep.data) {
                    if (randa.nextFloat() < perc) {
                        siftSample.addDataInstance(element);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * Loads the sample from a file.
     *
     * @param inFile Input file in Lowe's key format.
     * @param perc Float value that is the percentage of features to take.
     */
    private void loadSIFTFromSIFTFile(File inFile, float perc) {
        // Extract data and append to the current sample.
        SIFTRepresentation imageSIFTrep;
        Random randa = new Random();
        try {
            imageSIFTrep = SiftUtil.importFeaturesFromSift(inFile);
            if (!imageSIFTrep.isEmpty()) {
                for (DataInstance element : imageSIFTrep.data) {
                    if (randa.nextFloat() < perc) {
                        siftSample.addDataInstance(element);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * Loads the sample from a directory.
     *
     * @param inDir Input directory.
     * @param perc Float value that is the percentage of features to take.
     */
    private void loadSIFTFromDirectory(File inDir, float perc) {
        File[] arffFiles = inDir.listFiles(new ARFFFileNameFilter());
        File[] keyFiles = inDir.listFiles(new KeyFileNameFilter());
        File[] subdirectories = inDir.listFiles(new DirectoryFilter());
        if (recursive) {
            for (File dir : subdirectories) {
                loadSIFTFromDirectory(dir, perc);
            }
        }
        for (File arffFile : arffFiles) {
            loadSIFTFromARFFFile(arffFile, perc);
        }
        for (File keyFile : keyFiles) {
            loadSIFTFromSIFTFile(keyFile, perc);
        }
    }

    /**
     * Loads the sample from the current target.
     *
     * @param perc Float value that is the percentage of features to take.
     */
    public void getSIFTFromTargetARFFs(float perc) {
        if (target.isDirectory()) {
            loadSIFTFromDirectory(target, perc);
        } else {
            loadSIFTFromARFFFile(target, perc);
        }
    }
}
